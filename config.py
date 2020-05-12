import os
from os.path import isfile, join, isdir, dirname, abspath, realpath
from os import listdir

paths = {}


# please change the paths to the corresponding files
paths['path_calibrations']        = '/workspace/CSAI_Track1/Calibrations'

paths['path_videos']              = '/workspace/single_camera_mot/datasets/AIC20_track1/Dataset_A'

paths['path_list_video_id']       = '/workspace/single_camera_mot/datasets/AIC20_track1/list_video_id.txt'
paths['path_track1_vid_stats']    = '/workspace/single_camera_mot/datasets/AIC20_track1/track1_vid_stats.txt'

paths['path_faster_RCNN_weigths'] = '/workspace/single_camera_mot/detector/data/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'

paths['results_path'] = '/workspace/CSAI_Track1/Results'

config = {}
# Running Parameters

# to run faster the algorithm please use the maximum number of gpus available
# the id as CUDA_VISIBLE_DEVICES=
config['gpus_available'] = [8,9,10,11,12,13,14,15]

# parallel processes in the same GPU (reduce this if you run in memory problems) we use 2
config['parallel_processes'] = 1

# how many images load per run
config['batch_loader_size'] = 10

# dataloader per group of videos
config['num_threads'] = 2

# Submit params
# minimal length trayectory to count as car
config['min_length'] = 70

# multiplier, mean car box size
config['truck_times_car'] = 3


if __name__ == '__main__':
    print(paths)
