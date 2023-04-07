<div align="center">

# ASKPA: Quantitative Summarization

</div>

This repository maintains the code, data, and model checkpoints for the paper *Aspect Sentiment-based Key Point Analysis for Quantitative Review Summarization*

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
Model checkpoints are saved and accessed under the [```code/model/```](/code/model) folder. We released two models 
for reproducibility and evaluation.

Model checkpoints can be downloaded from this [Google Drive link](https://drive.google.com/drive/folders/1kIEsac0e819rX63PmENPfTctWWww1mIC?usp=sharing).

The `model/roberta-large-pretrained-yelp.rar` contains the checkpoint for the language model adapted to business reviews domain,
by pretraining on the Yelp dataset using the Masked LM task. The checkpoint must be located under the [```code/model/```](/model) folder

The `model/roberta-large-yelp-pretrained-contrastive-10-epochs-2023-02-15_23-31-51.rar` contains the checkpoint for the trained KP Matching model of ASKPA
used for our evaluation. The checkpoint must be located under the [```code/model/siamese-models```](code/model/siamese-models). 
Newly trained models should be found in the under the same folder.


## Code
For reproducing the experiment, the following notebooks should be executed:
- The [```contrastive_examples_construction_stage_1.ipynb```](/code/contrastive_examples_construction_stage_1.ipynb) notebook contains the code that extract and process raw data from the Yelp dataset 
to prepare for the construction of contrastive examples.
- The [```contrastive_examples_construction_stage_2.ipynb```](/code/contrastive_examples_construction_stage_2.ipynb) notebook contains the code to construct contrastive examples to prepare for training.
- The [```data_preparation.ipynb```](/code/data_preparation.ipynb) notebook contains the code that prepare and transform the data into desired input for training the siamese model
- The [```training.ipynb```](/code/training.ipynb) notebook contains the code to train the KP Matching model of ASKPA
- The [```evaluation.ipynb```](/code/evaluation.ipynb) notebook contains the code for model inference and evaluating the model