A. Transformer MLM and NSP Training

Reference : https://github.com/jamescalam/transformers/blob/main/course/training/08_mlm_and_nsp_training.ipynb

https://www.youtube.com/watch?v=1gN1snKBLP0

1. Install some module gradio app and weights and biases for experiment tracking.

2. Reproduce the code for understanding of Transformer MLM and NSP.

    a. Import module torch, Bert tokenizer, Bert for Next Sentence Prediction and Weights and Biases.

    b. Modify data into mix of non-random sentences, and random sentences for NSP. 

    c. Creates a 'bag of random sentences that we can pull from when selecting a random sentence B.

    d. Create a 50/50 NSP training data.

    e. Tokenize the data using a truncation/padding max_length of 512.
       This results to out-of-memory using GPU. Change the max_length=256.
       inputs = tokenizer(sentence_a, sentence_b, return_tensors='pt', max_length=256, truncation=True, padding='max_length')

    f. Create a labels tensor for MLM, and the next_sentence_label for NSP. The next_sentence_label tensor is simply a LongTensor([label]).T 
    Labels tensors is simply a clone of the input_ids tensor before masking.

    g. Now mask the tokens in the input_ids tensor using the 15% probability for MLM - ensuring we don't mask CLS, SEP, or PAD tokens. Then take the indices of each True value within each vector. 

    h. Apply these indices to each row in input_ids, assigning each value at these indices a value of 103. 

    i. Now all inputs and labels are ready, so set up the inputs to be fed into model during training. Create a PyTorch dataset from our data.

    j. Initialize the data using the MeditationsDataset class and Dataloader to load the data into model training. 

    k. Setup the training loop setting and the GPU/CPU usage. Activate the training mode of our model, and initialize the optimizer (Adam with weighted decay - reduces chance of overfitting).

    l. Initialize the weights and biases for training 2 epochs. 
       Result : 
       Activation for is the next sentence -1.1045 and not the next sentence 2.5738. 

       output.logits
          next sentence  not next sentence 
       tensor([[-1.1045,  2.5738]], device='cuda:0', grad_fn=<AddmmBackward>)

    m. Do the prediction
       Prediction is 1 means the next sentence(relationship with sentence_a to sentence_b)

       torch.argmax(outputs.logits)
       tensor(1, device='cuda:0')

    
    n. Add the gradio app for testing. I created the function for showing the results of next sentence based on the model train of MLM_NSP.


