#!/bin/bash

#######################################################################
#                            Original data                            #
#######################################################################
TRAIN_DATA_PATH="data/train-v3.csv"

DEV_DATA_PATH="data/trial-v3.csv"
DEV_LABELS_PATH_ORIG="data/trial-v3.labels"

TEST_DATA_PATH="data/test.csv"

#######################################################################
#                          Preprocessed data                          #
#######################################################################
PREPROCESSED_DATA_DIR="data/preprocessed"

CLEAN_TRAIN_DATA_PATH="$PREPROCESSED_DATA_DIR/train.csv"
TRAIN_LABELS_PATH="$PREPROCESSED_DATA_DIR/train_labels.csv"

CLEAN_DEV_DATA_PATH="$PREPROCESSED_DATA_DIR/dev.csv"
DEV_LABELS_PATH="$PREPROCESSED_DATA_DIR/dev_labels.csv"

CLEAN_TEST_DATA_PATH="$PREPROCESSED_DATA_DIR/test.csv"
TEST_LABELS_PATH="$PREPROCESSED_DATA_DIR/test_labels.csv"

#######################################################################
#                            No-emoji data                            #
#######################################################################
CLEAN_TRAIN_NO_EMOJIS_DATA_PATH="$PREPROCESSED_DATA_DIR/train_no_emojis.csv"
CLEAN_DEV_NO_EMOJIS_DATA_PATH="$PREPROCESSED_DATA_DIR/dev_no_emojis.csv"
CLEAN_TEST_NO_EMOJIS_DATA_PATH="$PREPROCESSED_DATA_DIR/test_no_emojis.csv"

################################################################################
#                           No-Triggerword data 
################################################################################
CLEAN_TRAIN_NO_TRIGGERWORD_DATA_PATH="$PREPROCESSED_DATA_DIR/train_no_triggerword.csv"
CLEAN_DEV_NO_TRIGGERWORD_DATA_PATH="$PREPROCESSED_DATA_DIR/dev_no_triggerword.csv"
CLEAN_TEST_NO_TRIGGERWORD_DATA_PATH="$PREPROCESSED_DATA_DIR/test_no_triggerword.csv"

function preprocess () {
    INFILE=$1
    OUTFILE=$2

    # The first awk call filters out those examples that do not contain
    # [#TRIGGERWORD#] The second one replaces keywords with a more standard
    # notation and adds spaces around them to make the job easier for the
    # tokenizer

    cat $INFILE | awk '{if ( $0 ~ /#TRIGGERWORD#/ ) {print $0}}' \
                | awk '{$1="";
                        gsub("\\[#TRIGGERWORD#\\]", " __TRIGGERWORD__ ", $0); 
                        gsub("@USERNAME", " __USERNAME__ ", $0); 
                        gsub("\\[NEWLINE\\]", " __NEWLINE__ ", $0); 
                        gsub("http://url.removed", " __URL__ ", $0);
                        print $0}' \
                | ./utils/twokenize.py > $OUTFILE
}

if [ ! -d $PREPROCESSED_DATA_DIR ]; then
    mkdir $PREPROCESSED_DATA_DIR
    echo "Created dir $PREPROCESSED_DATA_DIR"
fi

echo

################################################################################
#                                    TRAIN                                     #
################################################################################


echo "Preprocessing $TRAIN_DATA_PATH"
# The first awk call filters out those examples that do not contain [#TRIGGERWORD#]
# The second one replaces keywords with a more standard notation and adds spaces
# around them to make the job easier for the tokenizer

preprocess $TRAIN_DATA_PATH $CLEAN_TRAIN_DATA_PATH

cat $TRAIN_DATA_PATH | awk '{if ( $0 ~ /#TRIGGERWORD#/ ) {print $1}}' > $TRAIN_LABELS_PATH

echo "Created $CLEAN_TRAIN_DATA_PATH and $TRAIN_LABELS_PATH"

echo

echo "Creating Train dataset with no Triggerwords"
cat $CLEAN_TRAIN_DATA_PATH | awk '{gsub("__TRIGGERWORD__", "", $0); print $0}' > $CLEAN_TRAIN_NO_TRIGGERWORD_DATA_PATH
echo

echo "Creating Train dataset with no emoji"
cat $CLEAN_TRAIN_DATA_PATH | ./utils/remove_emojis.py > $CLEAN_TRAIN_NO_EMOJIS_DATA_PATH
echo "Created $CLEAN_TRAIN_NO_EMOJIS_DATA_PATH"

echo "################################"

################################################################################
#                                     DEV                                      #
################################################################################


echo "Preprocessing $DEV_DATA_PATH"
# Remove fake labels from dev data
awk '{$1=""; print $0}' $DEV_DATA_PATH > tmp.csv
# Then paste real labels
paste $DEV_LABELS_PATH_ORIG tmp.csv > tmp2.csv

rm tmp.csv
mv tmp2.csv tmp.csv

# Then we repeat the same process we did for the train data
preprocess tmp.csv $CLEAN_DEV_DATA_PATH

cat tmp.csv | awk '{if ( $0 ~ /#TRIGGERWORD#/ ) {print $1}}' > $DEV_LABELS_PATH
rm tmp.csv

echo "Created $CLEAN_DEV_DATA_PATH and $DEV_LABELS_PATH"

echo

echo "Creating Dev dataset with no Triggerwords"
cat $CLEAN_DEV_DATA_PATH | awk '{gsub("__TRIGGERWORD__", "", $0); print $0}' > $CLEAN_DEV_NO_TRIGGERWORD_DATA_PATH
echo

echo "Creating Dev dataset with no emoji"
cat $CLEAN_DEV_DATA_PATH | ./utils/remove_emojis.py > $CLEAN_DEV_NO_EMOJIS_DATA_PATH
echo "Created $CLEAN_DEV_NO_EMOJIS_DATA_PATH"

echo "################################"
################################################################################
#                                     TEST                                     #
################################################################################


echo "Preprocessing $TEST_DATA_PATH"

preprocess $TEST_DATA_PATH $CLEAN_TEST_DATA_PATH

cat $TEST_DATA_PATH | awk '{if ( $0 ~ /#TRIGGERWORD#/ ) {print $1}}' > $TEST_LABELS_PATH
echo "Created $CLEAN_TEST_DATA_PATH and $TEST_LABELS_PATH"

echo

echo "Creating Test dataset with no Triggerwords"
cat $CLEAN_TEST_DATA_PATH | awk '{gsub("__TRIGGERWORD__", "", $0); print $0}' > $CLEAN_TEST_NO_TRIGGERWORD_DATA_PATH
echo

echo "Creating Test dataset with no emoji"
cat $CLEAN_TEST_DATA_PATH | ./utils/remove_emojis.py > $CLEAN_TEST_NO_EMOJIS_DATA_PATH
echo "Created $CLEAN_TEST_NO_EMOJIS_DATA_PATH"

