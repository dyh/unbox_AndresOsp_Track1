import os
from os import listdir
from os.path import isfile, join, isdir

import time

import numpy as np
import pandas as pd

import yaml

import torch
from torch.utils.data import DataLoader

import tempfile

from PIL import Image, ImageDraw

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from Countor_NN import Countor_NN
from Calibration_loader import Calibration_loader
from Countor import Countor

from config import paths, config

import argparse

class VideoPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data):
        super(VideoPipe, self).__init__(batch_size, num_threads, device_id, seed=120)
        self.input = ops.VideoReader(device="gpu", filenames = data, sequence_length=1,
                                     shard_id=0, num_shards=1, random_shuffle=False, initial_fill=10)
    def define_graph(self):
        output = self.input(name="Reader")
        return output

def run_counter(config_file):

    # read file to know witch videos to process
    job = pd.read_csv(config_file)  

    # create Neural network
    countor_net = Countor_NN()
    countor_net.eval()
    countor_net.cuda()
    countor_net.load_state_dict(torch.load(paths['path_faster_RCNN_weigths'],map_location=lambda storage, loc: storage))

    for i, sub_job in job.iterrows():

        # create video loader
        pipeline = VideoPipe(batch_size=config['batch_loader_size'], 
                             num_threads=config['num_threads'], 
                             device_id=0, 
                             data=sub_job.path)
        pipeline.build()
        #
        dali_iter = DALIGenericIterator(pipeline,['image'], 
                                        pipeline.epoch_size("Reader"),
                                        fill_last_batch = False, 
                                        last_batch_padded = True)

        print("Start video: ",sub_job['name'],"Number of frames:",pipeline.epoch_size("Reader"))

        # load the calibration for this video
        calibration = Calibration_loader(str(sub_job.cam_config))

        # create counter object
        Counter = Countor(countor_net, calibration, 0)

        #video = cv2.VideoWriter('text.avi', 0, 1, (1280,960))

        multiplier = int(pipeline.epoch_size("Reader")/10)
        checkpoints = []
        for i in range(0,10):
            checkpoints.append(i*multiplier)

        num_frames = 0
        for i, video_data in enumerate(dali_iter):
            for image in video_data[0]['image']:
                image =  image.permute((0, 3, 1, 2)).contiguous().float().div(255)
                Counter.step(image)
                num_frames += 1
                if num_frames in checkpoints:
                    print(sub_job['name'],100*int(num_frames/pipeline.epoch_size("Reader")))


        # load the results as a DataFrame
        reform = {(outerKey, innerKey): values for outerKey, innerDict in  Counter.results.items() for innerKey, values in innerDict.items()}
        results =pd.DataFrame(reform,index=['left','top','x2','y2','conf']).T
        results.reset_index(inplace=True)
        results.rename(columns={"level_0": "ID", "level_1": "frame"},inplace=True)

        # correct data to original
        results['width']  = results['x2'] - results['left'] 
        results['height'] = results['y2'] - results['top'] 

        # put in format the restuls
        counts = pd.DataFrame.from_dict(Counter.countor_restults, orient='index',columns=['frame_id','movement_id','x1','y1','x2','y2','length'])
        counts.reset_index(inplace=True)
        counts.rename(columns={'index':'ID'},inplace=True)
        counts['video_id']=int(sub_job.ID)

        # path to save the resutls
        path_this_experiment = join(paths['results_path'],str(int(sub_job['ID'])))
        if not os.path.exists(path_this_experiment):
            os.makedirs(path_this_experiment)

        # save counts
        count_file=join(path_this_experiment,'counts.json')
        counts.to_json(count_file, orient='index')

        # save tracks
        tracks_file=join(path_this_experiment,'tracks.json')
        resultsinformat = results.loc[:,['frame','ID','left','top','width','height','conf']]
        resultsinformat.to_json(tracks_file, orient='index')

        # save parameters
        parameters_file = join(path_this_experiment,'parameters.yml')
        with open(parameters_file, 'w') as file:
            yaml.dump(calibration.calibration, file)
            
        # save parameters
        parameters_file = join(path_this_experiment,'parameters2.yml')
        with open(parameters_file, 'w') as file:
            yaml.dump(countor_net.get_parameters(), file)

        # save images
        width_line = 2
        epsulonx = [3,0]
        epsulony = [0,3]
        color = (255,0,0)

        for id_mov,df_id in counts.groupby(by='movement_id'):
            image_original = calibration.ROI_COLOR_image.copy()
            draw = ImageDraw.Draw(image_original)
            for j,row in df_id.iterrows():
                # draw vector init final position
                draw.line(tuple(row.values[3:7]),fill=tuple((np.random.rand(4)*255).astype(int)),width=width_line)
                # draw a red x in the final point
                draw.line(tuple(np.append(row.values[5:7]+epsulonx,row.values[5:7]-epsulonx)),fill=color,width=width_line)
                draw.line(tuple(np.append(row.values[5:7]+epsulony,row.values[5:7]-epsulony)),fill=color,width=width_line)

            del draw
            image_original.save(join(path_this_experiment,'lines_'+str(id_mov)+'.png'))

        print(sub_job['name'],"DONE")


    return True



# executes when your script is called from the command-line
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', help="Name of the file that has the videos to run")
    args = parser.parse_args()
    
    run_counter(args.file) 

