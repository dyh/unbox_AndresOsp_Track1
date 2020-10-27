# Unbox AI
- unbox opensource projects and products of Artificial Intelligence

### welcome to subscribe my channel
- [youtube channel](https://youtube.com/channel/UCAebg3DDFtidQJ0Jp20kyaw)
- [bilibili channel](https://space.bilibili.com/326361150)


## unbox project
- Implementation of Countor: count without bells and whistles


## video
- youtube
- video


## intro
- vehicle counting, vehicle tracking
- top teams in the 2020 AI City Challenge 
- Track 1: Multi-Class Multi-Movement Vehicle Counting


## system requirements
- ubuntu 18.04
- python >= 3.6
- cuda 10.2


## setup environments
1. clone source codes

    ```$ git clone https://github.com/dyh/unbox_AndresOsp_Track1.git```

2. enter project directory

    ```$ cd unbox_AndresOsp_Track1```

3. create a python virtual environment

    ```$ python3 -m venv venv```

4. activate the virtual environment

    ```$ source venv/bin/activate```

5. upgrade pip

    ```$ python -m pip install --upgrade pip```

6. install requirements package

    1. install other packages
    
        ```$ pip install -r requirements.txt```
    
    2. install NVIDIA DALI (based on cuda and pytorch)
    
        and you could choose other version at [here](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html)
    
        ```$ pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100```


## execute the program
1. download the weigths from the pytorch model zoo

    ```wget https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth```

2. detect the "./unbox_test/input/MVI_40855.mp4" video file of this project

    ```$ python unbox.py```

3. the output results are saved in "./unbox_test/output" directory for some image files

    0.png, 1.png, 2.png ... n.png

4. we can use ffmpeg to merge these images into one video file

    ```$ ffmpeg -f image2 -i ./unbox_test/output/%d.png ./unbox_test/output.mp4```

## sample dataset
- the test video [MVI_40855.mp4](/unbox_test/input/MVI_40855.mp4) is made up of images in the directory "MVI_40855" in the [DETRAC-test-data.zip](http://detrac-db.rit.albany.edu/Data/DETRAC-test-data.zip) file.
- download dataset: http://detrac-db.rit.albany.edu/download
- introduction of dataset: http://smart-city-sjsu.net/AICityChallenge/data.html


----

# AI开箱
- 人工智能开源项目和产品开箱

### 欢迎订阅我的频道
- [bilibili频道](https://space.bilibili.com/326361150)
- [youtube频道](https://youtube.com/channel/UCAebg3DDFtidQJ0Jp20kyaw)


## 开箱项目
- Implementation of Countor: count without bells and whistles


## 视频
- bilibili
- video


## 简介
- 车辆计数, 车辆追踪
- top teams in the 2020 AI City Challenge 
- Track 1: Multi-Class Multi-Movement Vehicle Counting


## 系统需求
- ubuntu 18.04
- python >= 3.6
- cuda 10.2


## 环境配置
1. 下载代码

    ```$ git clone https://github.com/dyh/unbox_AndresOsp_Track1.git```

2. 进入目录

    ```$ cd unbox_AndresOsp_Track1```

3. 创建python虚拟环境

    ```$ python3 -m venv venv```

4. 激活虚拟环境

    ```$ source venv/bin/activate```

5. 升级pip

    ```$ python -m pip install --upgrade pip```

6. 安装软件包

    1. 安装其他包
    
        ```$ pip install -r requirements.txt```

    2. 安装 NVIDIA DALI (基于 cuda 和 pytorch)
    
        你也可以在 [这里](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html) 选择其他版本
    
        ```$ pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100```


## 运行程序
1. 从pytorch model zoo下载weigths文件

    ```wget https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth```

2. 对项目中的 ./unbox_test/input/MVI_40855.mp4 视频文件进行检测

    ```$ python unbox.py```

3. 输出结果为图片文件，保存在 ./unbox_test/output 目录

    0.png, 1.png, 2.png ... n.png

4. 可以使用 ffmpeg 将图片文件合并为视频文件

    ```$ ffmpeg -f image2 -i ./unbox_test/output/%d.png ./unbox_test/output.mp4```

## 样本数据集
- 测试视频 [MVI_40855.mp4](/unbox_test/input/MVI_40855.mp4) 是由 [DETRAC-test-data.zip](http://detrac-db.rit.albany.edu/Data/DETRAC-test-data.zip) 文件中的 MVI_40855 目录下图片组成。
- 数据集下载 http://detrac-db.rit.albany.edu/download
- 数据集说明 http://smart-city-sjsu.net/AICityChallenge/data.html


----

# below is the origin README file

----
forked from [AndresOsp/Track1](https://github.com/AndresOsp/Track1)


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