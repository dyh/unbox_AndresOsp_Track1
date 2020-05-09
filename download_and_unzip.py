import urllib.request
import os
from os import listdir
from os.path import isfile, join, isdir
from pathlib import Path
from zipfile import ZipFile

from pathlib import Path

# tricky way to get the main path
parent_path  = Path(__file__).parent.absolute()
path_dataset = os.path.dirname(os.path.dirname(parent_path))
print(path_dataset)

# make dataset paths
name_dataset = 'AIC20_track1'
path = join(path_dataset,'datasets',name_dataset)

if not isdir(join(path_dataset,'datasets')):
    os.makedirs(join(path_dataset,'datasets'))

if not isdir(path):
    os.makedirs(path)

# urls
AIC20_track1_zip  = 'AIC20_track1_vehicle_counting.zip'

#url_efficiency_base ="http://www.aicitychallenge.org/wp-content/uploads/2020/03/efficiency_base.zip"
#AIC20_track2_zip  = 'AIC20_track2_reid.zip'
#AIC20_track22_zip = 'AIC20_track2_reid_simulation.zip'
#AIC20_track3_zip  = 'AIC20_track3_MTMC.zip'
#AIC20_track4_zip  = 'AIC20_track4_anomaly.zip'

url_base = 'http://www.aicitychallenge.org/wp-content/uploads/Shuo/2020/'+AIC20_track1_zip

# files 
path_AIC20_track1_zip  = join(path,AIC20_track1_zip)

print(path_AIC20_track1_zip)

# download dataset
if not isfile(path_AIC20_track1_zip):
    print('Downloading dataset from: ',url_base)
    # unzip 
    myCmd = 'wget -P '+path+' '+url_base
    myCmd = os.popen(myCmd).read()
    print(myCmd)


if not isdir(join(path,'AIC20_track1')):
    # unzip 
    print('Unzip the file: ',path_AIC20_track1_zip)
    myCmd = 'unzip '+path_AIC20_track1_zip+' -d '+path
    myCmd = os.popen(myCmd).read()
    print(myCmd)

if isdir(join(path,'AIC20_track1')):
    myCmd = 'mv '+join(path,'AIC20_track1')+'/* '+path
    myCmd = os.popen(myCmd).read()
    print(myCmd)

