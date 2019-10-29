# AVA_features

by William Veal Phan (william.veal-phan@etu.utc.fr)
using data from CIT and UTC

## Overview 

This project aims to train a model to extract features as described in Google's AVA dataset on small clips of video, focusing more particularly on person to person interactions interactions.

## Use

Define an AVAGenerator from a directory which should have the following structure :

```
dir/
    videos/
        vid1.mp4
        vid2.mp4
        ...
    labels/
        lab1.csv
         ...
```
with each lab#.csv begin the labels in a csv format for the corresponding video file.

This data can be consumed to train a model with input shape similar to the one in the models.py file (only one for now).

## TO DO
    - fix performances issues for regular PC
    - clean notebook example
    - add deeper models (the "deep_lstm" has only 2 lstm layers arranged in a sequencial model)