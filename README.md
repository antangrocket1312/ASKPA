<div align="center">

# ASKPA: Quantitative Summarization

</div>

This repository maintains the code, data, and model checkpoints for the paper *Aspect Sentiment-based Key Point Analysis for Quantitative Review Summarization*

## Dataset
We released the training and evaluation datasets of ASKPA. Datasets can be accessed under the ```data/``` folder, 
following the [```training/```](/data/training) and [```evaluation/```](/data/evaluation) subdirectories for each dataset

Files in each folder:
* ```.pkl```: data in .pkl format, accessible via Pandas library.
* ```.csv```: data in .csv format.

## Model checkpoints
Model checkpoints are saved and accessed under the [```code/model/```](/code/model) folder. We released two models 
for reproducibility and evaluation.

We made the two models available under the Hugggingface Hub as:
- [```quangantang/roberta-large-pretrained-yelp```](https://huggingface.co/quangantang/roberta-large-pretrained-yelp): The checkpoint for the language model adapted to business reviews domain,
by pretraining on the Yelp dataset using the Masked LM task.
- [```quangantang/roberta-large-askpa-matching ```](https://huggingface.co/quangantang/roberta-large-askpa-matching): the checkpoint for the trained KP Matching model of ASKPA 
used for our evaluation.

Newly trained models should be found in the under the [```code/model/siamese-models```](code/model/siamese-models) folder.


## Code
For reproducing the experiment, the following notebooks should be executed:
- The [```data_preparation.ipynb```](/code/data_preparation.ipynb) notebook contains the code that prepare and transform the data into desired input for training the siamese model
- The [```training.ipynb```](/code/training.ipynb) notebook contains the code to train the KP Matching model of ASKPA
- The [```evaluation.ipynb```](/code/evaluation.ipynb) notebook contains the code for model inference and evaluating the model