from collections import deque

import numpy as np
import torch

from intersection_class import counter_classification

class Countor:
    """The main tracking file, here is where magic happens."""

    def __init__(self, obj_detect, calibration, device):
        
        self.device = device
        
        # area minimal
        obj_detect.det_min_area = calibration.calibration['detection_min_area_val']
        obj_detect.tck_min_area = calibration.calibration['tracks_min_area_val']
        
        # area maximal
        obj_detect.det_max_area = calibration.calibration['detection_max_area_val']
        
        # thresh score
        #print(self.roi_heads.score_thresh)
        obj_detect.det_score_thresh = calibration.calibration['detection_object_thresh']
        obj_detect.tck_score_thresh = calibration.calibration['tracks_object_thresh']
        
        # nms thresh
        #print(self.roi_heads.nms_thresh)
        obj_detect.det_nms_thresh = calibration.calibration['detection_nms_thresh']
        obj_detect.tck_nms_thresh = calibration.calibration['tracks_nms_thresh']
        #self.empty_var = torch.empty(0, device=device)
        
        # porcentage ROI in
        obj_detect.det_min_ROI_in = calibration.calibration['detection_ROI_in']
        obj_detect.tck_min_ROI_in = calibration.calibration['tracks_ROI_in']
                
        # networks
        self.obj_detect = obj_detect
        
        # ROI image
        self.ROI_tensor = calibration.ROI_tensor.to(device)
        
        # Movement classification
        self.Movement_classification = counter_classification(calibration.movements)
        
        # Countor
        self.min_length_trajectory = calibration.calibration['min_length_trajectory']
        
        # motion model parameters
        self.motion_model_cfg  = calibration.calibration['motion_model']
      
        # set the values to 
        self.reset()

    def reset(self, hard=True):
        self.tracks = []
        self.inactive_tracks = []

        if hard:
            self.track_num = 0
            self.results = {}
            self.countor_restults = {}
            self.im_index = 0

    def tracks_to_inactive(self, tracks):
        self.tracks = [t for t in self.tracks if t not in tracks]
        for t in tracks:
            t.pos = t.last_pos[-1]          
            vector   = t.get_vector().cpu().numpy()
            movement = self.Movement_classification.classify(vector[:-1])
            self.countor_restults[t.id.cpu().numpy().item()] =np.concatenate([np.array([self.im_index]).astype(int),movement,vector])
        self.inactive_tracks += tracks

    def add(self, new_det_pos, new_det_scores):
        """Initializes new Track objects and saves them."""
        num_new = new_det_pos.size(0)
        for i in range(num_new):
            self.tracks.append(Track(
                new_det_pos[i].view(1, -1),
                new_det_scores[i],
                torch.tensor([self.track_num + i]).cuda(),
                self.motion_model_cfg['n_steps'] if self.motion_model_cfg['n_steps'] > 0 else 1
            ))
        self.track_num += num_new
    
    def get_pos(self):
        """Get the positions of all active tracks."""
        if len(self.tracks) == 1:
            pos = self.tracks[0].pos
            ids = self.tracks[0].id
        elif len(self.tracks) > 1:
            pos = torch.cat([t.pos for t in self.tracks], 0)
            ids = torch.cat([t.id for t in self.tracks], 0)
        else:
            pos = torch.empty(0, device=self.device)
            ids = torch.empty(0, device=self.device)
        return pos,ids

    def motion_step(self, track):
        """Updates the given track's position by one step based on track.last_v"""
        if self.motion_model_cfg['center_only']:
            center_new = get_center(track.pos) + track.last_v
            track.pos = make_pos(*center_new, get_width(track.pos), get_height(track.pos)).to(self.device)
        else:
            track.pos = track.pos + track.last_v

    def motion(self):
        """Applies a simple linear motion model that considers the last n_steps steps."""
        for t in self.tracks:
            last_pos = list(t.last_pos)

            # avg velocity between each pair of consecutive positions in t.last_pos
            if self.motion_model_cfg['center_only']:
                vs = [get_center(p2).to(self.device) - get_center(p1).to(self.device) for p1, p2 in zip(last_pos, last_pos[1:])]
            else:
                vs = [p2 - p1 for p1, p2 in zip(last_pos, last_pos[1:])]

            t.last_v = torch.stack(vs).mean(dim=0)
            self.motion_step(t)

                    
    def step(self, image):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.
        """
        for t in self.tracks:
            # add current position to last_pos list
            t.last_pos.append(t.pos.clone())

        # apply motion model
        if len(self.tracks):
            # apply motion model
            if self.motion_model_cfg['enabled']:
                self.motion()
                self.tracks = [t for t in self.tracks if t.has_positive_area()]
                
        pos, ids =self.get_pos()
        
        # Run the neural network
        det_boxes, det_scores, det_labels, tck_boxes, tck_scores, tck_labels, tck_ids = self.obj_detect(image, 
                                                                                                        [pos],
                                                                                                        [ids],
                                                                                                 [self.ROI_tensor])

        det_boxes = det_boxes[0].detach()
        det_scores = det_scores[0].detach()
        det_labels = det_labels[0].detach()
        tck_boxes = tck_boxes[0].detach()
        tck_scores = tck_scores[0].detach()
        tck_labels = tck_labels[0].detach()
        tck_ids = tck_ids[0].detach()
        
        # update position and score for the current tracks
        tracks_to_inactive_list = []
        if len(self.tracks):
            for i in range(len(self.tracks) - 1, -1, -1):
                if self.tracks[i].id in tck_ids:
                    index = tck_ids == self.tracks[i].id
                    index = index.nonzero().squeeze(1)
                    
                    self.tracks[i].pos   = tck_boxes[index].view(1, -1)
                    self.tracks[i].score = tck_scores[index]
                else:
                    # if the index is not in the ids set to inactive
                    tracks_to_inactive_list.append(self.tracks[i])

        # set the intective tracks  
        self.tracks_to_inactive(tracks_to_inactive_list)
        
        # Create new tracks
        if det_boxes.nelement() > 0:
            self.add(det_boxes, det_scores)
                

        ####################
        # Generate Results #
        ####################
        for t in self.tracks:
            if t.id.cpu().numpy().item() not in self.results.keys():
                self.results[t.id.cpu().numpy().item()] = {}
            self.results[t.id.cpu().numpy().item()][self.im_index] = np.concatenate([t.pos[0].cpu().numpy(), np.array([t.score.cpu()])])
        
        ####################
        # Prepare next frame #
        ####################
        self.im_index += 1
        self.last_image = image[0]   
        
    def get_results(self):
        return self.results


class Track(object):
    """This class contains all necessary for every individual track."""

    def __init__(self, pos, score, track_id, mm_steps):
        self.id = track_id
        self.pos = pos
        self.init_pos = pos
        self.score = score
        self.ims = deque([])
        self.last_pos = deque([pos.clone()], maxlen=mm_steps + 1)
        self.last_v = torch.Tensor([])


    def has_positive_area(self):
        return self.pos[0, 2] > self.pos[0, 0] and self.pos[0, 3] > self.pos[0, 1]
        
    def get_vector(self):
        x1y1 = get_center(self.init_pos)
        x2y2 = get_center(self.pos)
        dist = torch.dist(x1y1, x2y2,2).unsqueeze(0)
        return torch.cat([x1y1,x2y2,dist])

    
def get_center(pos):
    x1 = pos[0, 0]
    y1 = pos[0, 1]
    x2 = pos[0, 2]
    y2 = pos[0, 3]
    return torch.Tensor([(x2 + x1) / 2, (y2 + y1) / 2]).cuda()


def get_width(pos):
    return pos[0, 2] - pos[0, 0]


def get_height(pos):
    return pos[0, 3] - pos[0, 1]


def make_pos(cx, cy, width, height):
    return torch.Tensor([[
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2
    ]]).cuda()