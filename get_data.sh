#!/bin/bash

USER=$1

cd data
wget --user=$USER --ask-password http://implicitemotions.wassa2018.com/data/protected/train-v2.csv.gz
wget http://implicitemotions.wassa2018.com/data/unprotected/trial-v2.csv.gz
wget http://implicitemotions.wassa2018.com/data/unprotected/trial-v2.labels.gz

gunzip train-v2.csv.gz
gunzip trial-v2.csv.gz
gunzip trial-v2.labels.gz

cd ..
