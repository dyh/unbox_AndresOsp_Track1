import pandas as pd
from os.path import join

def read_info_files(paths):

    path_list_video_id = paths['path_list_video_id'] 
    path_track1_vid_stats = paths['path_track1_vid_stats'] 
    path_videos = paths['path_videos'] 
    # read ids and stats
    list_video_id = pd.read_csv(path_list_video_id,sep=' ',header=None,names=['ID','vid_name'])
    track1_vid_stats = pd.read_csv(path_track1_vid_stats,sep="\t")

    # generate dataframe with the information to run the algorithm
    track1_vid_stats.set_index('vid_name',inplace=True)
    track1_vid_stats['ID'] = list_video_id.set_index('vid_name')['ID'].astype('int32')
    track1_vid_stats.sort_values('frame_num',ascending=False,inplace=True)
    track1_vid_stats['name'] = [i.split('.')[0] for i in track1_vid_stats.index]
    track1_vid_stats['cam_config'] = [i.split('_')[1] for i in track1_vid_stats['name']]
    track1_vid_stats.reset_index(inplace=True)

    def generate_path(row):
        return join(path_videos,row['name'], row['vid_name'])+" "+str(row['ID'])+"\n"

    track1_vid_stats['path'] = track1_vid_stats.apply(generate_path, axis=1)
    
    return track1_vid_stats