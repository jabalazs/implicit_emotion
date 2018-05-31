#!/bin/bash

USER=$1

cd data
wget --user=$USER --ask-password http://implicitemotions.wassa2018.com/data/protected/train.csv.gz
wget http://implicitemotions.wassa2018.com/data/unprotected/trial.csv.gz
wget http://implicitemotions.wassa2018.com/data/unprotected/trial.labels.gz

gunzip train.csv.gz
gunzip trial.csv.gz
gunzip trial.labels.gz

cd ..
