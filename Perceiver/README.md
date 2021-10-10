
A. Perceiver ml model for classification in Keras.

ReadMe:

1. Reproduce the Sample code for understamding of Perceiver Implementation SOTA technology for image classification

      Perceiver use data augmentation, Feedforward network, patch creation and encoding of patch layer. To build the Perceiver model you need to implement Cross-attention, Transformer module.

2. Run the code using the CIFAR-100 dataset with colab GPU runtime, 10 epochs. 

      Results : 1 epoch took 4Hrs and didnt complete the one epoch run.

      Epoch 1/10
        404/704 [================>.............] - ETA: 4:44:34 4Hrs - loss: 4.4256 - acc: 0.0418 - top5-acc: 0.1474

3. Use Weights and Biases to run the code with GPU runtime, 50 and 10 epochs.

      Hyperparameters use first run : 

      Image size: 64 X 64 = 4096

      Patch size: 2 X 2 = 4 

      Patches per image: 1024

      Elements per patch (3 channels): 12

      Latent array shape: 256 X 256

      Data array shape: 1024 X 256

    Results : The hyperparameters needed to be optimize to use the GPU compute because of huge allocation of tensor with shape[64,256,8,256]. 

        ResourceExhaustedError:  OOM when allocating tensor with shape[64,256,8,256] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc
	    [[node perceiver/model_1/multi_head_attention_3/einsum_3/Einsum (defined at <ipython-input-15-05efc0397541>:89) ]]
        Hint: If you want to see a list of allocated tensors 4Hrs when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info. This isn't available when running in Eager mode.
        [Op:__inference_train_function_25330]

        Function call stack:
        train_function
4. Change the hyperparameters to a lower dimension 

      Image size: 32 X 32 = 1024

      Patch size: 4 X 4 = 16 
      
      Patches per image: 64
      
      Elements per patch (3 channels): 48
      
      Latent array shape: 64 X 64
      
      Data array shape: 64 X 64

      Results : 
      The model achieved Test accuracy: 55.79%

      Test top 5 accuracy: 95.1% running only for 10 epochs.

Conclusion : Smaller data array parameters gives higher accuracy compared with bigger data array parameters. 

Reference : 

https://keras.io/examples/vision/perceiver_image_classification/

https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/perceiver_image_classification.ipynb

https://github.com/wandb/client/archive/feature/code-save.zip


B. Perceiver IO model code in video, audio implementation 

1. Reproduce the sample code under the Apache license Version 2.0 to understand the implementation in video audio.

2. Install dependencies for Colab and Wandb, Imports needed modules including for Weights and Bias.

3. Fetch the open source videos from UCF101 dataset.

4. Load the video and audio from UCF.

5. Define the Kinetics 700 classes of videos.

6. Visualize the sample video audio. 

   Results : video of a lady putting eyeshadow makeup.

7. Define the Model Video autoencoder.

8. Apply the reconstruction model to video.

9. Load the parameters, and define the labels of video from Kinetics 700 labels.

10. Autoencode the entire video, one chunk at a time.
 
    Results : Using Wandb and GPU cannot autoencode the entire video.
    The colab crashed.

11. Visualize the reconstructed video.

    This will not show the entire video since the autoencode did not finish the running of code.
