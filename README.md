<div align="center">

# ASKPA: Quantitative Review Summarization

</div>

This repository maintains the code, data, and model checkpoints for the paper *Aspect-sentiment contrastive learning for 
quantitative review summarization*

## Dataset
We released the training and evaluation datasets of ASKPA. Datasets can be accessed under the ```data/``` folder, 
following the [```training/```](/data/training) and [```evaluation/```](/data/evaluation) subdirectories for each dataset.

Additionally, we provide the  [```yelp/```](/data/yelp) subdirectory that contains the raw, unprocessed and preprocessed data from Yelp
to allow reproducibility and extensibility. This folder can be downloaded
from this [Google Drive link](https://drive.google.com/drive/folders/1kIEsac0e819rX63PmENPfTctWWww1mIC?usp=sharing), 
under the `data/` directory.

Files in each folder:
* ```.pkl```: data in .pkl format, accessible via Pandas library.
* ```.csv```: data in .csv format.
* ```.jsonl```: data in .jsonl format (only for Yelp raw data).

## Model checkpoints
Model checkpoints are saved and accessed under the [```code/model/```](/code/model) folder. We released models trained under different settings (e.g. in-category/out-of-category)
for reproducibility and evaluation.

Model checkpoints can be downloaded from this [Google Drive link](https://drive.google.com/drive/folders/1XvjLh3IrpfCxnPoxphId0DYTQB3Eca2Q?usp=sharing).

We release three pretrained checkpoints for reproducibility of ASKPA. All checkpoints must be located under The checkpoint must be located 
under the [```code/model/```](/model) directory.
- `model/roberta-large-pretrained-yelp.zip` The model checkpoint of the RoBERTa-large model adapted to Yelp business reviews
by pretraining on the Masked LM task. For reproducibility, it can be utilized to fine-tune new KP Matching models for review summarization.
- `model/ASKPA.zip` The model checkpoint of ASKPA's contrastive KP Matching learning model, trained with data in different settings 
and business categories of reviews for evaluation.
Each model checkpoint is located in the respective ```{setting}/{category}/``` folder, while ```setting``` can either be **in-category** or **out-of-category**.
Simply place ASKPA folder into [```code/model/```](/model) from the working directory to reproduce evaluation results in the paper.
Any newly fine-tuned models can also be found in the under the same ```{setting}/{category}/``` folder.
- `model/roberta-large-finetuned-yelp-argument-quality-WA.zip` The model checkpoint of the argument quality ranking model fine-tuned on the Yelp-pretrained RoBERTa-large model, 
using ArgQ-14kPairs dataset. The model is used at the first stage of the KP Extraction process to select high-quality KPs that can be used to construct contrastive exmaples to train ASKPA.

## Code
For reproducing the ASKPA training and evaluation, we provide the following notebooks:
-  [```contrastive_examples_data_preprocessing.ipynb```](/code/contrastive_examples_data_preprocessing.ipynb) The notebook contains the code to pre-process, sample and select good data entries from 
the Yelp dataset to later construct contrastive examples in ```contrastive_examples_data_construction.ipynb```
-  [```contrastive_examples_data_construction.ipynb```](/code/contrastive_examples_data_construction.ipynb) The notebook contains the code to construct contrastive examples for training the ASKPA model.
-  [```ASKPA_training_preparation.ipynb```](/code/ASKPA_training_preparation.ipynb) The notebook contains the code to prepare and transform the training data into desired input to ASKPA's siamese model in in-category/out-of-category settings
-  [```ASKPA_training.ipynb```](/code/ASKPA_training.ipynb) The notebook contains the code to train the KP Matching model of ASKPA in in-category/out-of-category settings
-  [```ASKPA_evaluation.ipynb```](/code/ASKPA_evaluation.ipynb) The notebook contains the code for inference and evaluating the ASKPA model
-  [```ASKPA¬c_evaluation.ipynb```](/code/ASKPA¬c_evaluation.ipynb) The notebook contains the code to conduct evaluation on ASKPA¬c (the ablation study of ASKPA without contrastive learning)