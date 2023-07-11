<div align="center">

# ASKPA: Quantitative Review Summarization

</div>

This repository maintains the code, data, and model checkpoints for the paper *Aspect-sentiment contrastive learning for 
quantitative review summarization*

## Dataset
We released the training and evaluation datasets of ASKPA. Datasets can be accessed under the ```data/``` folder, 
following the [```training/```](/data/training) and [```evaluation/```](/data/evaluation) subdirectories for each dataset.

Additionally, we provide the  [```yelp/```](/data/yelp) subdirectory that contains the raw and preprocessed data from Yelp
to allow reproducibility and extensibility. This data can be downloaded
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

The `model/roberta-large-pretrained-yelp.rar` contains the checkpoint for the language model adapted to Yelp reviews,
by pretraining using the Masked LM task. The checkpoint must be located under the [```code/model/```](/model) folder. 
For reproducibility, it can be utilized to fine-tune new KP Matching models for review summarization

The `model/ASKPA.zip`the checkpoints of ASKPA's contrastive KP Matching leaning model, trained under different settings, 
and is used for our evaluation.
Each model checkpoint is located in the respective ```{setting}/{category}/``` folder, while ```setting``` can either be **in-category** or **out-of-category**.
Simply place ASKPA folder into [```code/model/```](/model) from the working directory to reproduce evaluation results in the paper.
Any newly fine-tuned models can also be found in the under the same ```{setting}/{category}/``` folder.

## Code
For reproducing the experiment, the following notebooks should be executed:
- The [```contrastive_examples_data_preparation.ipynb```](/code/contrastive_examples_data_preparation.ipynb) notebook contains the code that extract and process raw data from the Yelp dataset 
to prepare for the construction of contrastive examples.
- The [```contrastive_examples_data_construction.ipynb```](/code/contrastive_examples_data_construction.ipynb) notebook contains the code to construct contrastive examples to prepare for training.
- The [```data_preparation.ipynb```](/code/data_preparation.ipynb) notebook contains the code that prepare and transform the data into desired input for training the siamese model in in-category/out-of-category settings
- The [```ASKPA_training.ipynb```](/code/ASKPA_training.ipynb) notebook contains the code to train the KP Matching model of ASKPA in in-category/out-of-category settings
- The [```ASKPA_evaluation.ipynb```](/code/ASKPA_evaluation.ipynb) notebook contains the code for model inference and evaluating the model in-category/out-of-category settings