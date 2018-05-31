#!/bin/bash

PREPROCESSED_DATA_DIR="data/preprocessed"

TRAIN_DATA_PATH="data/train.csv"
DEV_DATA_PATH="data/trial.csv"

CLEAN_TRAIN_DATA_PATH="$PREPROCESSED_DATA_DIR/train.csv"
TRAIN_LABELS_PATH="$PREPROCESSED_DATA_DIR/train_labels.csv"
CLEAN_DEV_DATA_PATH="$PREPROCESSED_DATA_DIR/dev.csv"

CLEAN_TRAIN_NO_EMOJIS_DATA_PATH="$PREPROCESSED_DATA_DIR/train_no_emojis.csv"
CLEAN_DEV_NO_EMOJIS_DATA_PATH="$PREPROCESSED_DATA_DIR/dev_no_emojis.csv"


mkdir $PREPROCESSED_DATA_DIR
echo "Created dir $PREPROCESSED_DATA_DIR"

echo "Copying dev labels"
cp -v data/trial.labels $PREPROCESSED_DATA_DIR/dev_labels.csv

echo

echo "Preprocessing $TRAIN_DATA_PATH"
cat $TRAIN_DATA_PATH | awk '{$1=""; gsub("\\[#TRIGGERWORD#\\]", " __TRIGGERWORD__ ", $0); 
                                  gsub("@USERNAME", " __USERNAME__ ", $0); 
                                  gsub("\\[NEWLINE\\]", " __NEWLINE__ ", $0); 
                                  gsub("http://url.removed", " __URL__ ", $0); print $0}' \
                   | ./utils/twokenize.py > $CLEAN_TRAIN_DATA_PATH

cat $TRAIN_DATA_PATH | awk '{print $1}' > $TRAIN_LABELS_PATH

echo "Created $CLEAN_TRAIN_DATA_PATH and $TRAIN_LABELS_PATH"

echo

echo "Preprocessing $DEV_DATA_PATH"
cat $DEV_DATA_PATH | awk '{$1=""; gsub("\\[#TRIGGERWORD#\\]", " __TRIGGERWORD__ ", $0); 
                                  gsub("@USERNAME", " __USERNAME__ ", $0); 
                                  gsub("\\[NEWLINE\\]", " __NEWLINE__ ", $0); 
                                  gsub("http://url.removed", " __URL__ ", $0); print $0}' \
                   | ./utils/twokenize.py > $CLEAN_DEV_DATA_PATH
echo "Created $CLEAN_DEV_DATA_PATH"

echo

echo "Removing emojis from train"
cat $CLEAN_TRAIN_DATA_PATH | ./utils/remove_emojis.py > $CLEAN_TRAIN_NO_EMOJIS_DATA_PATH
echo "Created $CLEAN_TRAIN_NO_EMOJIS_DATA_PATH"

echo

echo "Removing emojis from dev"
cat $CLEAN_DEV_DATA_PATH | ./utils/remove_emojis.py > $CLEAN_DEV_NO_EMOJIS_DATA_PATH
echo "Created $CLEAN_DEV_NO_EMOJIS_DATA_PATH"
