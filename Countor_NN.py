

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
        
        
    # detections roi head
    def detections_processing(self, images, features, original_image_sizes):
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

        # split the boxes scores and labels for each image for post processing
        det_boxes_list  = det_boxes.split(boxes_per_image, 0)
        det_scores_list = det_scores.split(boxes_per_image, 0)
        det_labels_list = det_labels.split(boxes_per_image, 0)

        det_all_boxes = []
        det_all_scores = []
        det_all_labels = []

        for boxes, scores, labels, image_shape, original_im_shape in zip(det_boxes_list, 
                                                                         det_scores_list, 
                                                                         det_labels_list, 
                                                                         images.image_sizes, 
                                                                         original_image_sizes):
            
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes 
            inds = torch.nonzero(scores > self.roi_heads.score_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            nms(boxes, scores, countor_temp.roi_heads.nms_thresh)
            keep = box_ops.nms(boxes, scores, self.roi_heads.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.roi_heads.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            boxes = resize_boxes(boxes, image_shape, original_im_shape)

            det_all_boxes.append(boxes)
            det_all_scores.append(scores)
            det_all_labels.append(labels)

        return det_all_boxes,det_all_scores,det_all_labels


    def tracking_processing(self,boxes, images, features, original_image_sizes):
        device = list(self.parameters())[0].device
        # resize all the given box to be aligned
        tck_proposals = [resize_boxes(box, or_size, size) for box, or_size, size in zip(tempo_boxes, 
                                                                                        original_image_sizes, 
                                                                                        images.image_sizes) if box.nelement()]

        if tck_proposals:
            tck_box_features = countor_temp.roi_heads.box_roi_pool(features, tck_proposals, images.image_sizes)
            tck_box_features = countor_temp.roi_heads.box_head(tck_box_features)
            tck_class_logits, tck_box_regression = countor_temp.roi_heads.box_predictor(tck_box_features)
            tck_boxes = countor_temp.roi_heads.box_coder.decode(tck_box_regression, tck_proposals)
            tck_scores = F.softmax(tck_class_logits, -1)

            boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in tempo_boxes]

            tck_scores, tck_labels = tck_scores[:,countor_temp.selected_classes].max(1)
            tck_boxes = tck_boxes[:,countor_temp.selected_classes]
            tck_boxes = torch.cat([tck_boxes[idx][i].unsqueeze(0)for idx, i in enumerate(tck_labels)])

            tck_boxes_list  = tck_boxes.split(boxes_per_image, 0)
            tck_scores_list = tck_scores.split(boxes_per_image, 0)
            tck_labels_list = tck_labels.split(boxes_per_image, 0)
        else:
            tck_boxes_list  = torch.empty(0, device=device)
            tck_scores_list = torch.empty(0, device=device)
            tck_labels_list = torch.empty(0, device=device)
        
        return tck_boxes_list,tck_scores_list,tck_labels_list
           
    def forward(self, images, boxes):
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
            
        # get the list of detections
        det_all_boxes,det_all_scores,det_all_labels = detections_processing(images, features, original_image_sizes)
        # get the list of tracks
        tck_boxes_list,tck_scores_list,tck_labels_list = tracking_processing(boxes, images, features, original_image_sizes)
        
        return det_all_boxes, det_all_scores, det_all_labels, tck_boxes_list, tck_scores_list, tck_labels_list
