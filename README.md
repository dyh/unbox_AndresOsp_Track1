# Implementation of Countor: count without bells and whistles

This repository contains our implementation for 2020 AICity Challenge, and we achieve second place in Track 1: Multi-Class Multi-Movement Vehicle Counting.

## Abstract:
The effectiveness of an Intelligent transportation system (ITS) relies on the understanding of the vehicles behaviour.  Different approaches are proposed to extract the attributes of the vehicles as Re-Identification (ReID) or multi-target single camera tracking (MTSC). The analysis of those attributes leads to the behavioural tasks as multi-target multi-camera tracking (MTMC) and Turn-counts (Count vehicles that go through a predefined path). In this work, we propose a novel approach to Turn-counts which uses a MTSC and a proposed path classifier.  The proposed method is evaluated on CVPR AI City Challenge 2020. Our algorithm achieves the second place in Turn-counts with a score of 0.9346.

## Get started:
* Clone this repository
* Download the weigths from the pytorch model zoo:
```wget https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth```
* Download and extract the dataset from [AI City Challenge](https://www.aicitychallenge.org)
    * The dataset must contain:
        * A Folder with all the videos
        * The list_video_id.txt file witch gives the ID of each video
        * The track1_vid_stats.txt file with the number of frames of each video
        
* Open the `config.py` file:
    * Set the paths for the corresponding files
    * Set the GPU numbers that will be used to run the algorithm
    * If you run out of memory or problems with the processing power of the GPU please set parallel_processes to 1

* Then just run 'python run.py'

* When the program finish the results are in the Results folder.


## Dependencies
* pytorch 
* torchvision
* [nvidia-dali](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html)
* numpy
* pandas
* Pillow
* pyyaml



## Citation  
```
@InProceedings{Ospina_2020_CVPR_Workshops,
author = {Ospina, Andres and Torres, Felipe},
title = {Countor: Count Without Bells and Whistles},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2020}
}
```

## Authors:
* Andres OSPINA
* Felipe TORRES

## Used repositories
We used code from the following repositories:

* [Tracking without bells and whistles](https://github.com/phil-bergmann/tracking_wo_bnw)
* [Longest Processing Time](https://github.com/sanathkumarbs/longest-processing-time-algorithm-lpt)

## License

See [LICENSE](LICENSE). Please read before use.
