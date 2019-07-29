# Question Classifier

### Requirements
* edict
* pytorch 


### References
* Text preprocessing file was used from one of my previous projects which was borrowed from another repository
[RonghangHu-CMN](https://github.com/ronghanghu/cmn/blob/master/util/text_processing.py)


### Dataset 
* I couldn't find a question, non question dataset as such, so was looking around if I could prepare my own 
dataset. I tried [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) and [Amazon QA](http://jmcauley.ucsd.edu/data/amazon/qa/) but 
the datasets had their own problems.
* I then proceeded to download the [CoQA](https://stanfordnlp.github.io/coqa/) dataset, the dataset was 
a dialogues based dataset which is useful. In my implementation I broke down the story as not question and 
added the questions which were greater than 4 words. I ignore the answers as I the paragraph of the 
story resembled the test given to me.
* Number of training samples : 191367, Validation samples : 13638
* Basic preprocessing done in [read_data.py](read_data.py)


### Implementation details
* Set maximum words as 15, padded with zero
* Used GloVe 300d with 72704 words
* Finetuned the embedding
* Batch size - 10 - So that it isn't so heavy on my laptop - Can be increased


### Model
* Vanilla LSTM model, No attention - 99% train and val accuracy. 
* 1 layer - unidirectional (Can be changed to bidirectional in the main function)
* A dropout rate of 0.4


### Results
* The model gets trained pretty well in 1 epoch - upto 99% accuracy. So I set the max number of epochs as 2
* I checked the test results for a few samples and the model does make some errors, but overall it works well.
