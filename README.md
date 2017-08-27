# text_classification
This repo contains code for training Machine Learning models for text classification.

## Overview
There are two types of ML algorithms.

- traditional - logistic regression, support vector machines, mutinomial naive bayes with tf-idf features. (Note: mutinomail naive bayes uses tf features because of the independent assumption.)

- neural - Recurrent Neural Networks and Convolutional Neural Networks.

## Datasets
* AG's News Topic Classification Dataset

## Current Results
|   accuracy    |  RNN  | CNN  | LR  | SVM  | MNB  | NBSVM |
| ------------- | ----- | ---- | --- | ---- | ---- | ---   | 
|   AG's News   |  87.1% | 88.1%  | 92.3%| 92.6% | 91.7% | |

## Reference
Recurrent Neural Network (RNN): [Character-level Convolutional Networks for Text Classification](https://arxiv.org/pdf/1509.01626.pdf)

Convolutional Neural Network (CNN): [Convolutional Neural Networks for Sentence Classification, Kim 2014](https://arxiv.org/pdf/1408.5882.pdf) 

Naive Bayse SVM (NBSVM) : [Baselines and Bigrams: Simple, Good Sentiment and Topic Classification, Wang et al. 2012](https://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf)


