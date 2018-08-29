#!/bin/bash

USER=$1

cd data
wget --user=$USER --ask-password http://implicitemotions.wassa2018.com/data/protected/train-v3.csv.gz
wget http://implicitemotions.wassa2018.com/data/unprotected/trial-v3.csv.gz
wget http://implicitemotions.wassa2018.com/data/unprotected/trial-v3.labels.gz

wget http://implicitemotions.wassa2018.com/data/unprotected/test-text-labels.csv.gz
mv test-text-labels.csv.gz test.csv.gz

gunzip train-v3.csv.gz
gunzip trial-v3.csv.gz
gunzip trial-v3.labels.gz
gunzip test.csv.gz

cd ..
