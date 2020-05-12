import torch
import torch.nn.functional as F

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import resize_boxes
from torchvision.ops.boxes import clip_boxes_to_image, nms

class Countor_NN(FasterRCNN):

    def __init__(self):
        backbone = resnet_fpn_backbone('resnet50', False)
        super(Countor_NN, self).__init__(backbone, 91)
        # get cars trucks and bus classes
        self.selected_classes = [3,6,8]
        
        # area minimal
        self.det_min_area = 100
        self.tck_min_area = 100
        
        # area maximal
        self.det_max_area = 1000000
        
        # thresh score
        #print(self.roi_heads.score_thresh)
        self.det_score_thresh = 0.5
        self.tck_score_thresh = 0.05
        
        # nms thresh
        #print(self.roi_heads.nms_thresh)
        self.det_nms_thresh = 0.5
        self.tck_nms_thresh = 0.5
        #self.empty_var = torch.empty(0, device=device)
        
        # porcentage ROI in
        self.det_min_ROI_in = 0.3
        self.tck_min_ROI_in = 0.1
        
    def get_parameters(self):
        
        parameters = {}
        # area minimal
        parameters['det_min_area'] = self.det_min_area 
        parameters['tck_min_area'] = self.tck_min_area 
        
        # area maximal
        parameters['det_max_area'] = self.det_max_area 
        
        # thresh score
        #print(self.roi_heads.score_thresh)
        parameters['det_score_thresh'] = self.det_score_thresh 
        parameters['tck_score_thresh'] = self.tck_score_thresh 
        
        # nms thresh
        #print(self.roi_heads.nms_thresh)
        parameters['det_nms_thresh'] = self.det_nms_thresh 
        parameters['tck_nms_thresh'] = self.tck_nms_thresh 
        #self.empty_var = torch.empty(0, device=device)
        
        # porcentage ROI in
        parameters['det_min_ROI_in'] = self.det_min_ROI_in
        parameters['tck_min_ROI_in'] = self.tck_min_ROI_in
        
        return parameters
        
    @staticmethod
    def remove_small_boxes_area(boxes, min_size):
        
        # type: (Tensor, float)
        """
        Remove boxes which area is smaller than min_size.
        Arguments:
            boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format
            min_size (float): minimum size
        Returns:
            keep (Tensor[K]): indices of the boxes that have both sides
                larger than min_size
        """
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep = area >= min_size
        keep = keep.nonzero().squeeze(1)
        return keep

    @staticmethod
    def remove_big_boxes_area(boxes, max_size):
        
        # type: (Tensor, float)
        """
        Remove boxes which area is smaller than min_size.
        Arguments:
            boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format
            min_size (float): minimum size
        Returns:
            keep (Tensor[K]): indices of the boxes that have both sides
                larger than min_size
        """
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep = area <= max_size
        keep = keep.nonzero().squeeze(1)
        return keep
            
    @staticmethod
    def remove_boxes_out_roi(boxes, ROI_image, min_in_porcentage):
        
        # type: (Tensor, float)
        """
        Remove boxes which contains at least one side smaller than min_size.
        Arguments:
            boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format
            min_size (float): minimum size
        Returns:
            keep (Tensor[K]): indices of the boxes that have both sides
                larger than min_size
        """
        
        area_in =[ROI_image[box[1]:box[3],box[0]:box[2]].sum() for box in boxes.int()]
        area_in = torch.stack(area_in) 
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        porcentage_in = area_in/area.float()
        keep = porcentage_in >= min_in_porcentage
        keep = keep.nonzero().squeeze(1)
        return keep
    
    # detections roi head

    def detections_processing(self, images, features, ROI_images, tck_boxes, original_image_sizes):
        # rpn network
        det_proposals, proposal_losses = self.rpn(images, features)

        # roi heads to get boxes and scores
        det_box_features = self.roi_heads.box_roi_pool(features, det_proposals, images.image_sizes)
        det_box_features = self.roi_heads.box_head(det_box_features)
        det_class_logits, det_box_regression = self.roi_heads.box_predictor(det_box_features)
        det_boxes = self.roi_heads.box_coder.decode(det_box_regression, det_proposals)
        det_scores = F.softmax(det_class_logits, -1)


        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in det_proposals]

        # this gets the max score and gives the label
        # this mean just for 3 classes get just one
        det_scores, det_labels = det_scores[:,self.selected_classes].max(1)
        det_boxes  = det_boxes[:,self.selected_classes]
        det_boxes = torch.cat([det_boxes[idx][i].unsqueeze(0)for idx, i in enumerate(det_labels)])

        # split the boxes scores and labels for each image for post processing
        det_boxes_list  = det_boxes.split(boxes_per_image, 0)
        det_scores_list = det_scores.split(boxes_per_image, 0)
        det_labels_list = det_labels.split(boxes_per_image, 0)

        det_all_boxes = []
        det_all_scores = []
        det_all_labels = []
        

        for boxes, scores, labels, ROI_image, tck_boxes_b, image_shape, original_im_shape in zip(det_boxes_list, 
                                                                         det_scores_list, 
                                                                         det_labels_list,
                                                                         ROI_images,
                                                                         tck_boxes,
                                                                         images.image_sizes, 
                                                                         original_image_sizes):

            boxes = clip_boxes_to_image(boxes, image_shape)

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            
            # remove low scoring boxes 
            inds = torch.nonzero(scores > self.det_score_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove small boxes
            keep = self.remove_small_boxes_area(boxes, min_size=self.det_min_area)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            
            # remove too big boxes
            #keep = self.remove_big_boxes_area(boxes, max_size=self.det_max_area)
            #boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            
            # non-maximum suppression, independently done per class
            keep = nms(boxes, scores, self.det_nms_thresh)
            #boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            
            # keep only topk scoring predictions
            keep = keep[:self.roi_heads.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            
            boxes = resize_boxes(boxes, image_shape, original_im_shape)
            
            if boxes.nelement():
                keep = self.remove_boxes_out_roi(boxes, ROI_image, min_in_porcentage=self.det_min_ROI_in)
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                
            # filter detections in tracks  
            for tck_box in tck_boxes_b:
                temp_boxes  = torch.cat([tck_box.unsqueeze(0), boxes])
                temp_scores = torch.cat([torch.tensor([2.0]).to(boxes.device), scores])
                keep   = nms(temp_boxes, temp_scores, self.det_nms_thresh)
                keep   = keep[torch.ge(keep, 1)] - 1
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                if keep.nelement() == 0:
                    break

            
            det_all_boxes.append(boxes)
            det_all_scores.append(scores)
            det_all_labels.append(labels)

        return det_all_boxes, det_all_scores, det_all_labels


    def tracking_processing(self,boxes, boxes_ids, images , features, ROI_images, original_image_sizes):
        device = list(self.parameters())[0].device
        # resize all the given box to be aligned
        tck_proposals = [resize_boxes(box, or_size, size) for box, or_size, size in zip(boxes, 
                                                                                        original_image_sizes, 
                                                                                        images.image_sizes) if box.nelement()]
        
        tck_all_boxes  = []
        tck_all_scores = []
        tck_all_labels = []
        tck_all_ids = []
        
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in tck_proposals]
        
        if tck_proposals:
            tck_box_features = self.roi_heads.box_roi_pool(features, tck_proposals, images.image_sizes)
            tck_box_features = self.roi_heads.box_head(tck_box_features)
            tck_class_logits, tck_box_regression = self.roi_heads.box_predictor(tck_box_features)
            tck_boxes = self.roi_heads.box_coder.decode(tck_box_regression, tck_proposals)
            tck_scores = F.softmax(tck_class_logits, -1)

            tck_scores, tck_labels = tck_scores[:,self.selected_classes].max(1)
            tck_boxes = tck_boxes[:,self.selected_classes]
            tck_boxes = torch.cat([tck_boxes[idx][i].unsqueeze(0)for idx, i in enumerate(tck_labels)])

            tck_boxes_list  = tck_boxes.split(boxes_per_image, 0)
            tck_scores_list = tck_scores.split(boxes_per_image, 0)
            tck_labels_list = tck_labels.split(boxes_per_image, 0)
            
            for boxes, scores, labels, box_ids, ROI_image, image_shape, original_im_shape in zip(tck_boxes_list, 
                                                                         tck_scores_list, 
                                                                         tck_labels_list, 
                                                                         boxes_ids,
                                                                         ROI_images,
                                                                         images.image_sizes, 
                                                                         original_image_sizes):
            
                boxes = clip_boxes_to_image(boxes, image_shape)

                # batch everything, by making every class prediction be a separate instance
                boxes = boxes.reshape(-1, 4)
                scores = scores.reshape(-1)
                labels = labels.reshape(-1)
                box_ids = box_ids.reshape(-1)

                # remove low scoring boxes 
                keep    = torch.nonzero(scores > self.tck_score_thresh).squeeze(1)
                boxes   = boxes[keep]
                scores  = scores[keep]
                labels  = labels[keep]
                box_ids = box_ids[keep]

                # remove small boxes
                keep    = self.remove_small_boxes_area(boxes, min_size=self.tck_min_area)
                boxes   = boxes[keep]
                scores  = scores[keep]
                labels  = labels[keep]
                box_ids = box_ids[keep]
                
                # non-maximum suppression, independently done per class
                keep    = nms(boxes, scores, self.tck_nms_thresh)
                boxes   = boxes[keep]
                scores  = scores[keep]
                labels  = labels[keep]
                box_ids = box_ids[keep]
                # keep only topk scoring predictions

                boxes = resize_boxes(boxes, image_shape, original_im_shape)
                
                if boxes.nelement():
                    
                    keep = self.remove_boxes_out_roi(boxes, ROI_image, min_in_porcentage=self.tck_min_ROI_in)
                    boxes   = boxes[keep]
                    scores  = scores[keep]
                    labels  = labels[keep]
                    box_ids = box_ids[keep]

                tck_all_boxes.append(boxes)
                tck_all_scores.append(scores)
                tck_all_labels.append(labels)
                tck_all_ids.append(box_ids)
        else:
            tck_all_boxes.append(torch.empty(0, device=device))
            tck_all_scores.append(torch.empty(0, device=device))
            tck_all_labels.append(torch.empty(0, device=device))
            tck_all_ids.append(torch.empty(0, device=device))
        
        return tck_all_boxes,tck_all_scores,tck_all_labels, tck_all_ids
           
    def forward(self, images, boxes, boxes_ids, ROI_images):
        # inputs to device
        device = list(self.parameters())[0].device
        
        # get the original image size
        original_image_sizes = [img.shape[-2:] for img in images]
        
        targets = None
        # transform the images with the standart 
        images, targets = self.transform(images,targets)
        # get the features form the transformd images
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])
            
        # get the list of tracks
        tck_boxes, tck_scores, tck_labels, tck_ids = self.tracking_processing(boxes, 
                                                                              boxes_ids, 
                                                                              images, 
                                                                              features, 
                                                                              ROI_images, 
                                                                              original_image_sizes)
            
        # get the list of detections
        det_boxes, det_scores, det_labels = self.detections_processing(images, 
                                                                       features, 
                                                                       ROI_images, 
                                                                       tck_boxes, 
                                                                       original_image_sizes)
        
        return det_boxes, det_scores, det_labels, tck_boxes, tck_scores, tck_labels, tck_ids
