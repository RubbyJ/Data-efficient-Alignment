# Data-efficient-Alignment
This repository is an implementation of WACV 2021 paper 'Data-efficient Alignment of Multimodal Sequences by Aligning GradientUpdates and Internal Feature Distributions'.
[Arxiv Preprint](https://arxiv.org/abs/2011.07517)


# Contents
* [Prerequisites](#prerequisites)
* [Dataset & Feature Extraction](#dataset--feature-extraction)
* [Highligts](#highlights)
* [Training Example](#training-example)

## Prerequisites

- python == 3.6
- pytorch == 1.4.0
- tensorboard

## Dataset & Feature Extraction

### Dataset

Please refer to [@pelindogan/NeuMATCH](https://github.com/pelindogan/NeuMATCH) this repo for YouTube Movie Summaries(YMS) Dataset. 
`YMS/annotations_phrases.txt` shows the ground truth matched pairs.
We ask the authors for YMS data splits, which are provided in `YMS_Dataset` directory. 

### Feature Extraction

For each video clip, we extract features of the central frame using Faster-RCNN trained on the Visual Genome dataset. 
`VG Detector` of this repo [@nocaps-org/image-feature-extractors](https://github.com/nocaps-org/image-feature-extractors) is helpful and the model can be found in [@peteanderson80/bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention).

For a text snippet, we extract 768-dimensional sentence embedding from the [BERT](https://github.com/google-research/bert) - Base model. 

## Highlights

- [SBN](src/model/SBN.py) - an implementation of Sequence-wise Batch Normalization (not support Multi-GPU training now).
- [LARS](src/solver/larses.py) - a function for adding LARS to an Adam optimizer.

## Training Example

An example for training,
```
python do.py \
--lr 7 
--loss ls --lsr_epsilon 0.03 \
--adamlars --lars_coef 1e-3 \
--SBN \
--random_project \
--dataset yms \
--epochs 350
```



