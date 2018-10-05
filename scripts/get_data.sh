#!/bin/bash

USER=$1

cd data
################################################################################
#                           TRAIN / DEV / TEST data                            #
################################################################################

wget --user=$USER --ask-password http://implicitemotions.wassa2018.com/data/protected/train-v3.csv.gz
wget http://implicitemotions.wassa2018.com/data/unprotected/trial-v3.csv.gz
wget http://implicitemotions.wassa2018.com/data/unprotected/trial-v3.labels.gz

wget http://implicitemotions.wassa2018.com/data/unprotected/test-text-labels.csv.gz
mv test-text-labels.csv.gz test.csv.gz

gunzip train-v3.csv.gz
gunzip trial-v3.csv.gz
gunzip trial-v3.labels.gz
gunzip test.csv.gz

mkdir word_embeddings
cd word_embeddings
################################################################################
#                         Pre-trained GloVe embeddings                         #
################################################################################

# Uncomment these lines if you want to get GloVe embeddings
# wget http://nlp.stanford.edu/data/glove.840B.300d.zip
# unzip glove.840B.300d.zip

################################################################################
#                           Pre-trained ELMo Weights                           #
################################################################################
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5
