# Model checkpoints

Model checkpoints can be downloaded from this [Google Drive link](https://drive.google.com/drive/folders/1XvjLh3IrpfCxnPoxphId0DYTQB3Eca2Q?usp=sharing).

The `model/roberta-large-pretrained-yelp.rar` contains the checkpoint for the language model adapted to Yelp reviews,
by pretraining using the Masked LM task. The checkpoint must be located under the [```code/model/```](/model) folder. 
For reproducibility, it can be utilized to fine-tune new KP Matching models for review summarization

The `model/ASKPA.zip`the checkpoints of ASKPA's contrastive KP Matching leaning model, trained under different settings, 
and is used for our evaluation.
Each model checkpoint is located in the respective ```{setting}/{category}/``` folder, while ```setting``` can either be **in-category** or **out-of-category**.
Simply place ASKPA folder into [```code/model/```](/model) from the working directory to reproduce evaluation results in the paper.
Any newly fine-tuned models can also be found in the under the same ```{setting}/{category}/``` folder.