# Data-efficient-Alignment
This repository is an implementation of WACV 2021 paper 'Data-efficient Alignment of Multimodal Sequences by Aligning GradientUpdates and Internal Feature Distributions'.
[Arxiv Preprint](https://arxiv.org/abs/2011.07517)


# Contents
* [Prerequisites](#prerequisites)
* [Dataset & Feature Extraction](#dataset--feature-extraction)
  * [Dataset](#dataset)
  * [Feature Extraction](#feature-extraction)
* [Highligts](#highlights)
* [Quick Start](#quick-start)
  * [Downloading](#downloading)
  * [Evaluation Example](#evaluation-example)
  * [Training Example](#training-example)

## Prerequisites

- python == 3.6
- pytorch == 1.4.0
- tensorboard

Or 
`conda env create -f alignment.yml`.

## Dataset & Feature Extraction

### Dataset

The [YMS_dataset](YMS_dataset) directory provides the information for YouTube Movie Summaries(YMS) Dataset. 


### Feature Extraction

For each video clip, we extract features of the central frame using Faster-RCNN trained on the Visual Genome dataset. 
`VG Detector` of this repo [@nocaps-org/image-feature-extractors](https://github.com/nocaps-org/image-feature-extractors) is helpful and the model can be found in [@peteanderson80/bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention).

For a text snippet, we extract 768-dimensional sentence embedding from the [BERT](https://github.com/google-research/bert) - Base model. 


## Highlights

- [SBN](src/model/SBN.py) - an implementation of Sequence-wise Batch Normalization (not support Multi-GPU training now).
- [LARS](src/solver/larses.py) - a function for adding LARS to an Adam optimizer.


## Quick Start 

### Downloading
For the convenience of experience, we provide the processed data and 
a pretrained Model (RP+LARS+SBN) at [Google Cloud](https://drive.google.com/drive/folders/1sPh3FDQ3g_OtBJrw_jDL0FPUjskJmlKV?usp=sharing). 
Please put it under this directory. 

### Evaluation Example
After downloading the data and model, let's take an evaluation example on TEST.

```
CUDA_VISIBLE_DEVICES=0 python do.py \
--evaluate \
--SBN \
--random_project \
--dataset yms \
--where_best './data/RP_SBN_LARS.ckpt' 
```


### Training Example

An example for training,

```
CUDA_VISIBLE_DEVICES=0 python do.py \
--lr 7 
--loss ls --lsr_epsilon 0.03 \
--adamlars --lars_coef 1e-3 \
--SBN \
--random_project \
--dataset yms \
--epochs 350
```


[YMS]:https://github.com/RubbyJ/NeuMATCH
