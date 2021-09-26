
A. How to do Contrastive Learning SimCLR using Pytorch in Colaboratory

1. Clone the sample code from the github for reproducibilty, experimentation and understanding. Save it to gdrive. Refer github Code: https://github.com/thunderInfy/simclr
2. Run the SimCLR code in colab using the terminal command and make sure you are connected to CUDA for reproducibility.
3. Results using CUDA in Colab with not enough memory. Reuse the trained model save in results folder and imagenet-5-categories images.
4. Prepare the training datasets with class AugmentedDataset with color jitter and random resize crop.
5. Check the images if can be read properly for train and test set sample.
6. Load the augmented images with batch size of 250.
7. Plot the TSNE visualization
8. Perform Linear Classifier with visualization of training/test accuracy/losses.


B. How to do Contrastive Learning SimCLR using Tensorflow in Colaboratory
1. Clone the dataset Imagenet. Refer github Code: https://github.com/thunderInfy/imagenet-5-categories
2. Prepare the images for Train and test from image paths.
3. Create TensorFlow dataset and class for data augmentation with random crops, random flips, color jitter, gaussian blur and random apply.
4. Create a data augmentation pipeline.
5. Initialize the NT_Xentloss.
6. Perform the SimCLR training and visualization with early stopping to prevent overfitting.
