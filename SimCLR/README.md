
A. How to do Contrastive Learning SimCLR using Tensorflow in Colaboratory

Clone the dataset Imagenet. Refer github Code: https://github.com/thunderInfy/imagenet-5-categories

Prepare the images for Train and test from image paths.

Create TensorFlow dataset and class for data augmentation with random crops, random flips, color jitter, gaussian blur and random apply.

Create a data augmentation pipeline.

Initialize the NT_Xentloss.

Perform the SimCLR training and visualization with early stopping to prevent overfitting.
