# Dependencies
Library: Python=3.6, PyTorch>0.4.0,tensorboardX

# Implementation Details
To accelerate the training process, we trained the detection and segmentation modules separately. In particular, the weights of the detection module are frozen when training the segmentation module.

# Pretrained Weights
resnet50 weight,detection weight and segmentation weight can found in release.

# DataSets file structure
                   |->mask         
                   |->test
DataSets->kaggle-> |->val
                   |->train
