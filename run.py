import os
from os import listdir
from os.path import isfile, join, isdir

from multiprocessing import Pool   

import time

import numpy as np
import pandas as pd


from Countor_NN import Countor_NN
from utils.LongestProcessingTime import LPT
from utils.files_preprocessing import read_info_files


from config import paths, config

# executes when your script is called from the command-line
if __name__ == "__main__":



    # get start time
    start = time.time()

    # read the files from the dataset (list_video_id.txt and track1_vid_stats.txt) 
    videos_info = read_info_files(paths)

    print("[INFO] "+"The information of the videos to process is:")
    print(videos_info.drop(columns='path'))

    gpus_available = []

    for i in range(config['parallel_processes']):
        gpus_available.extend(config['gpus_available'])

    temp = LPT(videos_info)
    scheduled_jobs, loads = temp.run_for_custom_batch_size(len(gpus_available))

    if not isdir('tmp'):
        os.mkdir('tmp')

    if not isdir(paths['results_path']):
        os.mkdir(paths['results_path'])

    print("[INFO] "+"The algorithm group the videos in the following groups for each GPU:")
    name_file_to_sub = []
    for i,(df,loa,gpu) in enumerate(zip(scheduled_jobs,loads,gpus_available)):
        print(50*"_")
        print("GROUP: ",i+1)
        print(df.drop(columns='path'))
        name_file_to_sub.append(join('tmp',str(i))+".csv")
        df.to_csv(name_file_to_sub[-1],index=False)
        print("Total frames: ",loa )

    processes = []
    for file, gpu in zip(name_file_to_sub,gpus_available):
        processes.append("CUDA_VISIBLE_DEVICES="+str(gpu)+" python sub_run.py "+file)
        print(processes[-1])

    print("[INFO] "+"Processing the videos:")
    def run_process(process):                                                             
        os.system(process)  

    #pool = Pool()                                                        
    #pool.map(run_process, processes)    


    pool = Pool()                                                        
    pool.map(run_process, processes)    

    print("[INFO] "+"Classifing 4 wheel or truck")
    
    videos_info.sort_values('ID',inplace=True)

    dfs = []

    for i,row in videos_info.iterrows():

        path_this_experiment = join(paths['results_path'],str(int(row['ID'])))
        path_track = join(path_this_experiment,'tracks.json')
        df_tracks = pd.read_json(path_track,orient='index')
        df_tracks.sort_values(['ID'],inplace=True)
        df_tracks['area']=df_tracks['width']*df_tracks['height']

        path_counts = join(path_this_experiment,'counts.json')
        df_counts = pd.read_json(path_counts,orient='index')
        df_counts.sort_values(['ID'],inplace=True)

        mean_area = {}
        median_area = {}
        for i,df_i in df_tracks.groupby('ID'):
            mean_area[i] = df_i['area'].mean()
            median_area[i] = df_i['area'].median()

        df_counts.set_index('ID',inplace=True)
        new_row = []
        new_row2 = []
        for i,row in df_counts.iterrows():
            new_row.append(mean_area[int(i)])
            new_row2.append(median_area[int(i)])

        df_counts['mean']   =new_row
        df_counts= df_counts[df_counts['movement_id']>0]
        df_counts= df_counts[df_counts['length']>config['min_length']]
        df_counts['vehicle_class_id']=1

        trucks = []

        for i,df_i2 in df_counts.groupby('movement_id'):
            df_temp =     df_i2.copy()
            df_temp.sort_values('mean',inplace=True)
            take_80_p = int(len(df_temp.index)*0.9)
            std_80p = df_temp.iloc[:take_80_p]['mean'].std()
            mean_80p = df_temp.iloc[:take_80_p]['mean'].mean()
            tresh_big= config['truck_times_car']*mean_80p

            df_i3 =df_i2[df_i2['mean']>tresh_big]
            trucks+=list(df_i3.index)

        df_counts.loc[trucks,'vehicle_class_id']=2
        df_counts.reset_index(inplace=True)
        dfs.append(df_counts)
        
    print("[INFO] "+"Generating resutls")

    df_submission = pd.concat(dfs, ignore_index=True)
    resultsinformat = df_submission.loc[:,['video_id','frame_id','movement_id','vehicle_class_id']]
    resultsinformat.rename(columns={'video_id':'video_clip_id'},inplace=True)
    resultsinformat.to_csv(join(paths['results_path'] ,'Submission.txt'),index=False,header=False)

    print('Total_time:', time.time()-start)

    with open(join(paths['results_path'] ,'Total_time.txt'), "w") as text_file:
        text_file.write("Total_time: %s" % (time.time() - start))