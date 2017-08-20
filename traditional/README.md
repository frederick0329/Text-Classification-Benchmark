# Traditional Models

This folder contains code/models/logs for traditional models.

## Getting Started

You can simply run the following code to train a model from scratch.

For *logistic regression*

```
python traditional/train.py -classifier lr > traditional/logs/segment_level.log
```

The best model will be saved in the models folder.

The best model is obtained by grid search on the hyper parameters with cross-validation


## Usage

```
python train.py -h
```
