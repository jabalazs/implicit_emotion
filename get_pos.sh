#!/bin/bash

PREPROCESSED_DATA_DIR="data/preprocessed"

CLEAN_TRAIN_DATA_PATH="$PREPROCESSED_DATA_DIR/train.csv"
TRAIN_LABELS_PATH="$PREPROCESSED_DATA_DIR/train_labels.csv"

CLEAN_DEV_DATA_PATH="$PREPROCESSED_DATA_DIR/dev.csv"
DEV_LABELS_PATH="$PREPROCESSED_DATA_DIR/dev_labels.csv"

CLEAN_TEST_DATA_PATH="$PREPROCESSED_DATA_DIR/test.csv"
TEST_LABELS_PATH="$PREPROCESSED_DATA_DIR/test_labels.csv"


#######################################################################
#                                TRAIN                                #
#######################################################################


echo "Getting POS tags for $CLEAN_TRAIN_DATA_PATH with TwiboParser"

# this script will generate the file $CLEAN_TRAIN_DATA_PATH.tagged.pre
./utils/run_postagger.sh $CLEAN_TRAIN_DATA_PATH

cat $CLEAN_TRAIN_DATA_PATH.tagged.pre | ./utils/postprocess_postag_file.py > $CLEAN_TRAIN_DATA_PATH.tagged
rm $CLEAN_TRAIN_DATA_PATH.tagged.pre

# this script will generate two files:
# - $CLEAN_TRAIN_DATA_PATH.tagged.tokens
# - $CLEAN_TRAIN_DATA_PATH.tagged.pos
./utils/parse_twibo_output.py $CLEAN_TRAIN_DATA_PATH.tagged

echo
#######################################################################
#                                 DEV                                 #
#######################################################################
echo "Getting POS tags for $CLEAN_DEV_DATA_PATH with TwiboParser"

./utils/run_postagger.sh $CLEAN_DEV_DATA_PATH

cat $CLEAN_DEV_DATA_PATH.tagged.pre | ./utils/postprocess_postag_file.py > $CLEAN_DEV_DATA_PATH.tagged
rm $CLEAN_DEV_DATA_PATH.tagged.pre

python ./utils/parse_twibo_output.py $CLEAN_DEV_DATA_PATH.tagged

echo
#######################################################################
#                                TEST                                 #
#######################################################################
echo "Getting POS tags for $CLEAN_TEST_DATA_PATH with TwiboParser"
./utils/run_postagger.sh $CLEAN_TEST_DATA_PATH

cat $CLEAN_TEST_DATA_PATH.tagged.pre | ./utils/postprocess_postag_file.py > $CLEAN_TEST_DATA_PATH.tagged
rm $CLEAN_TEST_DATA_PATH.tagged.pre

./utils/parse_twibo_output.py $CLEAN_TEST_DATA_PATH.tagged
echo

