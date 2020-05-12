import os
from os import listdir
from os.path import isfile, join, isdir

import yaml

import numpy as np
import pandas as pd

import torch

from PIL import Image, ImageDraw

from config import paths, config

class Calibration_loader():
    """COUNTER Dataset.

    This dataloader is designed so that it can handle only one sequence, if more have to be
    handled one should inherit from this class.
    """

    def __init__(self, cam_config):
        """
        Args:
            seq_name (string): Sequence to take
            vis_threshold (float): Threshold of visibility of persons above which they are selected
        """
        # paths to the difernet folders and files
        self.path_config      = join(paths['path_calibrations'],cam_config)
        self.roi_csv_path     = join(self.path_config,'rois_new.csv')
        self.movements_path   = join(self.path_config,'movements.csv')
        self.calibration_path = join(self.path_config,'Counter_parameters.yml')
        self.image_cam        = join(self.path_config,'000001.jpg')
        
        self.device           = 0

        # check if the folders exist
        if not isfile(self.calibration_path):
            raise AssertionError('The path of the config file', self.calibration_path,' do not exist')
        else:
            with open(self.calibration_path) as file:
                self.calibration = yaml.load(file, Loader=yaml.FullLoader)
            
        # load ROI image
        if not isfile(self.roi_csv_path):
            raise AssertionError('The path of the ROI csv ', self.roi_csv_path,' do not exist')
        else:
            self.generate_ROI_images()
            
        # load movements data
        if not isfile(self.movements_path):
            raise AssertionError('The path of the calibration movements file', self.movements_path,' do not exist')
        else: 
            self.movements = pd.read_csv(self.movements_path)


    def generate_ROI_images(self):
        # read polygon points
        self.rois_polygon   = pd.read_csv(self.roi_csv_path)
        polygon =[tuple(i) for i in self.rois_polygon.values] 
        # read one image
        im_temp =Image.open(self.image_cam)
        # new image of roi
        img = Image.new('L', (im_temp.width, im_temp.height), 0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        self.ROI_array = np.array(img)
        
        self.ROI_tensor= torch.from_numpy(self.ROI_array).to(self.device)
       
        ROI = np.array(self.ROI_array)
        ROI[ROI==0] = 200
        ROI[ROI==1] = 255

        red_im = np.zeros(ROI.shape+(3,),dtype=np.uint8)
        red_im[:,:,0]=255

        ROI_image = Image.fromarray(ROI)
        RED_image = Image.fromarray(red_im)
        
        self.ROI_COLOR_image = Image.composite(im_temp, RED_image, ROI_image)
        #self.ROI_COLOR_image.save(join(self.path_config,'ROI.jpg'))