# Nerual Models

This folder contains code/models/logs for neural models.

## Getting Started

You can simply run the following code to train a model from scratch.

For *RNN*

```
python -W ignore -u neural/train_rnn.py > neural/logs/segment_level.log
```


For *CNN*

```
python -W ignore -u neural/train_cnn.py > neural/logs/segment_level.log
```

The best model will be saved in the models folder.

The best model is selected with a held out develop set.

## Usage

```
python train_rnn.py -h
```

```
python train_cnn.py -h
```
