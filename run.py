import os
from os import listdir
from os.path import isfile, join, isdir

import time

import numpy as np
import pandas as pd

from tqdm import tqdm
import tempfile

from Countor_NN import Countor_NN
from utils.LongestProcessingTime import LPT
from utils.files_preprocessing import read_info_files

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator

# Parameters
# what device to use
device_id = 0

# parameters that change the memory usage
optimized_batch_size = True
custom_batch_size = 10


# dataloader per group of videos
num_threads = 5

# save the images for analisys
save_images = True

# just to divide the printed information
DELIMITER1 = 80*"="
DELIMITER2 = 100*"_"

def run_algorithm(path_images,path_config,path_weights):
    
    # Dataloader
    Counter_sequence    = Countor_loader(path_images, path_config)
    Counter_data_loader = DataLoader(Counter_sequence, batch_size=1, shuffle=False)
    
    number_of_frames = len(Counter_sequence)
    multiplier = int(number_of_frames/10)
    checkpoints = []
    for i in range(0,10):
        checkpoints.append(i*multiplier)

    print("Start video: ",Counter_sequence.video_info['vid_name'],"Number of frames:",number_of_frames)
    
    # Faster RCNN 
    Counter_FRCNN = Countor_FRCNN()
    Counter_FRCNN.load_state_dict(torch.load(path_weights,map_location=lambda storage, loc: storage))
    Counter_FRCNN.eval()
    Counter_FRCNN.cuda()

    # Countor
    Counter = Countor(Counter_FRCNN, Counter_sequence)
    
    num_frames = 0
    Counter.reset()
    #for i, frame in enumerate(tqdm(Counter_data_loader,ncols=50,ascii=True,desc=Counter_sequence.video_info['vid_name'])):
    for i, frame in enumerate(Counter_data_loader):
        Counter.step(frame)
        num_frames += 1
        if num_frames in checkpoints:
            print(Counter_sequence.video_info['vid_name'],num_frames/number_of_frames)
                        
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
    counts['video_id']=Counter_sequence.video_info['ID']
    
    # path to save the resutls
    path_this_experiment = join(paths['ROOT_PATH'],'testing',str(Counter_sequence.video_info['ID']))
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
        yaml.dump(Counter_sequence.calibration, file)
    
    
    # save images
    if save_images:
        path_images = join(path_this_experiment,'counts.json')
        width_line = 2
        epsulonx = [3,0]
        epsulony = [0,3]
        color = (255,0,0)

        for id_mov,df_id in counts.groupby(by='movement_id'):
            image_original = Counter_sequence.ROI_COLOR_image.copy()
            draw = ImageDraw.Draw(image_original)
            for j,row in df_id.iterrows():
                # draw vector init final position
                draw.line(tuple(row.values[3:7]),fill=tuple((np.random.rand(4)*255).astype(int)),width=width_line)
                # draw a red x in the final point
                draw.line(tuple(np.append(row.values[5:7]+epsulonx,row.values[5:7]-epsulonx)),fill=color,width=width_line)
                draw.line(tuple(np.append(row.values[5:7]+epsulony,row.values[5:7]-epsulony)),fill=color,width=width_line)

            del draw
            image_original.save(join(path_this_experiment,'lines_'+str(id_mov)+'.png'))
            
    print(Counter_sequence.video_info['vid_name'],"DONE")
    
    return True

class VideoPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data):
        super(VideoPipe, self).__init__(batch_size, num_threads, device_id, seed=120)
        self.input = ops.VideoReader(device="gpu", file_list = data, sequence_length=1,
                                     shard_id=0, num_shards=1, random_shuffle=False, initial_fill=10)

    def define_graph(self):
        output = self.input(name="Reader")
        return output

# executes when your script is called from the command-line
if __name__ == "__main__":
    
    paths = {}
    # please change the paths to the corresponding files
    paths['path_videos']              = '/workspace/single_camera_mot/datasets/AIC20_track1/Dataset_A'
    paths['path_list_video_id']       = '/workspace/single_camera_mot/datasets/AIC20_track1/list_video_id.txt'
    paths['path_track1_vid_stats']    = '/workspace/single_camera_mot/datasets/AIC20_track1/track1_vid_stats.txt'
    paths['path_calibrations']        = ''
    paths['path_faster_RCNN_weigths'] = '/workspace/single_camera_mot/detector/data/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'
    

    videos_info = read_info_files(paths)

    print("[INFO] "+"The information of the videos to process is:")
    print(DELIMITER1)
    print(videos_info.drop(columns='path'))
    print(DELIMITER1)
    
    temp = LPT(videos_info)
    if optimized_batch_size:
        scheduled_jobs, loads = temp.find_best_batch_size()
        print("[INFO] "+"If your system runs out of memory please set optimized_batch_size to False and set a lower batch size at custom_batch_size")
    else:
        scheduled_jobs, loads = temp.run_for_custom_batch_size(custom_batch_size)
        print("[INFO] "+"You are currently running in a custom batch. If your system runs out of memory set a lower batch size at custom_batch_size")

    print("[INFO] "+"The batch size is:", len(loads))
    
    print("[INFO] "+"The algorithm group the videos in the following groups:")
    for i,(df,loa) in enumerate(zip(scheduled_jobs,loads)):
        print(DELIMITER1)
        print("GROUP: ",i+1)
        print(df.drop(columns='path'))
        print("Total frames: ",loa )
        print(DELIMITER1)
        print("[INFO] "+"Creating dataloader pipelines")
        

    text_files = []
    files_names = []
    pipelines = []

    for job in scheduled_jobs:
        temp_txt =''
        for i in list(job['path'].values):
            temp_txt += i
        temp_file = tempfile.NamedTemporaryFile()
        temp_file.write(str.encode(temp_txt))
        temp_file.flush()

        text_files.append(temp_file)
        files_names.append(temp_file.name)

        pipelines.append( VideoPipe(batch_size=100, num_threads=num_threads, device_id=device_id, data=temp_file.name))
        pipelines[-1].build()
        
    print("[INFO] "+"Creating dataloader")
    dali_iter = DALIGenericIterator(pipelines, 
                                    ['image','cam_id'], 
                                    max(loads), 
                                    fill_last_batch=False, 
                                    last_batch_padded=True, 
                                    dynamic_shape=True)
    
    
    current_batch = len(scheduled_jobs)+1

    for i, video_data in enumerate(tqdm(dali_iter)):
        pass
        #images = []
        #camera_id = []
        #for i,data in enumerate(video_data):
        #    images.append( data['image'][0][0].permute((2, 0, 1)).contiguous().float().div(255))
        #    camera_id.append( data['cam_id'][0][0])
        #if len(images)!=current_batch:
        #    print(len(images))
         #   current_batch=len(images)
    
    
    