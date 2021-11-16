1. Reptile MAML

   References : [keras reptile](https://keras.io/examples/vision/reptile/) and [sample colab](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/reptile.ipynb#scrollTo=XpJM2NGbyUhU)

   a. Import Modules
   b. Define the Hyperparameters.
   c. Prepare MNIST dataset for training.
   d. Visualize some examples from the dataset.
   e. Build the model
   f. Train the model
   g. Visualize the results.
   
   ![Screen Shot 2021-10-31 at 11 24 21 PM](https://user-images.githubusercontent.com/62075076/139630425-f25d7333-7df2-424b-bcac-e0a466a2de13.png)
   
 2. MMOE 
 
    References : [keras mmoe](https://github.com/drawbridge/keras-mmoe) and [census income dataset from UCI](https://bit.ly/2wLWmAY)
   
    a. Install the modules from the [requirements.txt](https://github.com/drawbridge/keras-mmoe/blob/master/requirements.txt).
   
    b. Clone the repository.
   
    c. Convert the python code [census_income.py](https://github.com/drawbridge/keras-mmoe/blob/master/census_income_demo.py) and [synthetic_demo.py](https://github.com/drawbridge/keras-mmoe/blob/master/synthetic_demo.py) to colab.
   
    d. Run the converted colab code with the correct path of dataset for census income and synthetic demo.
    
    e. Results of two colabs.
    
       ![Screen Shot 2021-11-09 at 11 08 07 AM](https://user-images.githubusercontent.com/62075076/140988820-df53dc20-8980-4be6-8bcf-b4a966762521.png)

       ![Screen Shot 2021-11-09 at 11 08 44 AM](https://user-images.githubusercontent.com/62075076/140988829-7d6665ed-189e-46ea-a913-b5943519a3c3.png)

 3. Avalanche Continual AI
 
    Reference : [Avalanche](https://github.com/ContinualAI/avalanche) and its [tutorials](https://avalanche.continualai.org/from-zero-to-hero-tutorial/01_introduction)
    
    a. Implement Omniglot Dataset using the same tutorial with MLP model and SplitMNIST.
    
       Result : Got an error in shape input. Change the Model input MLP and flatten the input but the error still persist.
       
       ![Screen Shot 2021-11-09 at 11 13 29 AM](https://user-images.githubusercontent.com/62075076/140989635-026dc717-bc28-4a80-b5cf-3fc3b621fcd6.png)
       
       
       ![Screen Shot 2021-11-09 at 11 13 50 AM](https://user-images.githubusercontent.com/62075076/140989795-c8f4d8cd-e070-4f71-98ec-1deecd3cf393.png)

       
    b. Run the MNIST Dataset for comparison with the Omniglot Dataset. Supposedly it should work the same since it both use same channels and pixel size.
    
       Result : Training runs up to four experiences.
       
       ![Screen Shot 2021-11-09 at 11 14 20 AM](https://user-images.githubusercontent.com/62075076/140989983-da57c3e9-e8e9-4f47-bbdc-84a7c08f20b0.png)

   


 
   
   
