# Question Classifier

### Requirements
* edict
* pytorch 


### References
* Text preprocessing file was used from one of my previous projects which was borrowed from another repository
[RonghangHu-CMN](https://github.com/ronghanghu/cmn/blob/master/util/text_processing.py)


### Dataset 
* [CoQA](https://stanfordnlp.github.io/coqa/) a dialogues based dataset. Not questions - story sentences and questions - the one greater than 4 words.
* Number of training samples : 191367, Validation samples : 13638
* Basic preprocessing done in [read_data.py](read_data.py)


### Implementation details
* Maximum words - 15, padded with zero
* GloVe 300d with 72704 words
* Finetuned embedding
* Batch size - 10

### Model
* Vanilla LSTM model, No attention - 99% train and val accuracy. 
* 1 layer - unidirectional 
* A dropout rate of 0.4


### Results
* The model gets trained pretty well in 1 epoch - upto 99% accuracy. I've set the max number of epochs as 2
