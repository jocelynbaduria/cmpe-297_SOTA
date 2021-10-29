A. Transformer MLM and NSP Training

Reference : 
https://github.com/jamescalam/transformers/blob/main/course/training/08_mlm_and_nsp_training.ipynb

https://www.youtube.com/watch?v=1gN1snKBLP0

1. Install some module gradio app and weights and biases for experiment tracking.

3. Reproduce the code for understanding of Transformer MLM and NSP.

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
    
B. Question and Answering
Reference :

https://arxiv.org/pdf/1810.04805.pdf

https://huggingface.co/transformers/model_doc/bert.html?highlight=bertforquestionanswering#transformers.BertForQuestionAnswering

https://colab.research.google.com/drive/1uSlWtJdZmLrI3FCNIlUHFxwAJiSu2J0-#scrollTo=6_mAnIPKaXyw

Pre-trained model gradio sample : https://github.com/kamalkraj/BERT-SQuAD

1. Install some module gradio app and weights and biases for experiment tracking, and hugging face transformers library.
2. Load Fine-Tuned BERT-large.
3. Ask a simple question.
4. Initialize the Weights and Biases for tracking of training.

![Screen Shot 2021-10-28 at 8 36 27 PM](https://user-images.githubusercontent.com/62075076/139371639-aaa4a609-5bd8-478d-aa9c-9f731574c48b.png)

6. Visualization of the Scores.
7. Use Pretrained model sample and Initialize Weights and biases for tracking of training.
8. Add gradio app for Testing
![Screen Shot 2021-10-27 at 6 59 12 PM](https://user-images.githubusercontent.com/62075076/139173156-42430f3a-8161-4ba2-a669-f6ab70580d1c.png)

C. GPT-2 Text Generation 

1. Install module gradio, wandb and transformer
2. Initialize  weights and biases (id = bert_pretrained)
![Screen Shot 2021-10-27 at 9 08 45 PM](https://user-images.githubusercontent.com/62075076/139185355-9c12d949-0175-4d2b-a7d3-f7447e7626ae.png)
4. Load the GPT-2 model and create a funtion for text generation
5. Add Gradio app Interface for Testing GPT-2.
![Screen Shot 2021-10-27 at 8 50 50 PM](https://user-images.githubusercontent.com/62075076/139185241-34fc5ad9-5be1-4746-879b-7e096665abfc.png)

D. Bart Text Summarizer

Reference: 

https://huggingface.co/facebook/bart-large-cnn

https://huggingface.co/google/pegasus-cnn_dailymail

https://towardsdatascience.com/building-nlp-web-apps-with-gradio-and-hugging-face-transformers-59ce8ab4a319

https://github.com/chuachinhon/gradio_nlp/blob/main/notebooks/2.0_gradio_parallel_summaries.ipynb

1. Import Libraries and install some module gradio, wandb and transformers.

2. Define text cleaning and Summarization functions.

3. Initialize Weights and Bias and use Hugging Face Pipeline to implement the pre-trained model facebook/bart-large-cnn and google/pegasus-cnn_dailymail.

![Screen Shot 2021-10-28 at 2 46 19 PM](https://user-images.githubusercontent.com/62075076/139341082-f2f90ee9-0c77-4a84-8b79-a0c1fd11f8ce.png)

4. Launch both model facebook and google text sumarizer for comparison using gradio App for testing.
![Screen Shot 2021-10-28 at 2 36 19 PM](https://user-images.githubusercontent.com/62075076/139340605-2518add4-988c-4d3f-8e50-0fda2ff51331.png)

E. Wav2Vec Audio Transcribing

Reference: 

https://huggingface.co/facebook/bart-large-cnn

https://huggingface.co/google/pegasus-cnn_dailymail

https://towardsdatascience.com/building-nlp-web-apps-with-gradio-and-hugging-face-transformers-59ce8ab4a319

https://github.com/chuachinhon/gradio_nlp/blob/main/notebooks/2.0_gradio_parallel_summaries.ipynb

1. Import Libraries and install some module gradio, wandb and transformers.

2. Initialize  weights and biases (id = asr_5)

![Screen Shot 2021-10-28 at 8 34 29 PM](https://user-images.githubusercontent.com/62075076/139371437-e3c27919-9a95-49b9-a474-f605fedb114f.png)

3. Load pre-trained Wav2Vec2 model and define the speech to text function. You can choose any model that will suits your needs from [here](https://huggingface.co/facebook/wav2vec2-base-960h)

4. Add Gradio app for testing.

![Screen Shot 2021-10-28 at 8 28 00 PM](https://user-images.githubusercontent.com/62075076/139371362-c4de1283-7921-4351-a3e5-3cc39debe673.png)


