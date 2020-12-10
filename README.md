# Project: Object detection

## Overview
The target of this homework is Instance segmentation.  
Training on Tiny PASCAL VOC dataset with 20 common object classes and trying to detect each segmentation.

## Hardware
The following specs were used to create the original solution.

* Ubuntu 18.04 LTS
* NVIDIA GeForce RTX 2080

## Download Official Image
You can download the training and testing dataset from Google Drive.  
https://drive.google.com/drive/folders/1fGg03EdBAxjFumGHHNhMrz2sMLLH04FK

## Installation
* Linux or macOS with Python ≥ 3.6
* PyTorch ≥ 1.5 and torchvision that matches the PyTorch installation.
* OpenCV is optional and needed by demo and visualization
* Numpy 1.19.2
* Tqdm 4.51.0
* Cuda 10.1
* Detectron 2
 
You need to create a dictory names 'checkpoints' to save chekpoint.  
Download Detectron2 by follow the github  
https://github.com/facebookresearch/detectron2
  
## Usage
Run train.py to start training.   
```
python train.py
```

Test the model  
```
python test.py
```

## Reference
http://host.robots.ox.ac.uk/pascal/VOC/   

###### tags: `Instance segmentation` `Deep learning` `NCTU CS` `Detectron`
