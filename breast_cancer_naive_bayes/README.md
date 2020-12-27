# Lab 4: Naive Bayes

**CSCI 360: Introduction to Artificial Intelligence**

## Introduction
The goal is to predict from 9 features whether breast cancer patients had recurrence 
events or not. The features are all categorical features, in the sense that they have 
levels. For example, the levels of node-caps are yes and no. The levels of breast-quad 
are left-up, left-low, right-up, right-low, and central. The dataset includes 201 
instances of no recurrence and 85 instances of recurrence. You are given training data 
that has 160 instances of the no-recurrence class and 68 instances of the recurrence 
class. Your task is to build a Na ̈ıve Bayes’ classifier that predicts recurrence of 
breast cancers.

In this lab you will be implementing the algorithm, but first you will
have to clean the data. The data is found in the [`data.npy`](./data.npy).

All the code you write should be in [`lab4.py`](./lab4.py) and you will be
under functions `preprocess_data` and `naive_bayes`. It is
important you don't change the parameters. You are provided with a
utility file and a test file. The utility file has functions provided
that will compute the `accuracy_score` and `load_data`.

It also contains the names of the features in the dataset in the order that they appear in the columns.

You are allowed to use `numpy` which is outlined by [`requirements.txt`](./requirements.txt). Note you are not allowed to use sklearn.


## Test File:
The test file can be run it will try to use the `preprocess_data` and `naive_bayes` as they are outline in the lab4 PDF.

The test file uses `load_data` this pulls a tuple from the data
