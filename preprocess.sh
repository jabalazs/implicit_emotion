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
TEST_LABELS_PATH="$PREPROCESSED_DATA_DIR/test_labels_fake.csv"

#######################################################################
#                            No-emoji data                            #
#######################################################################
CLEAN_TRAIN_NO_EMOJIS_DATA_PATH="$PREPROCESSED_DATA_DIR/train_no_emojis.csv"
CLEAN_DEV_NO_EMOJIS_DATA_PATH="$PREPROCESSED_DATA_DIR/dev_no_emojis.csv"
CLEAN_TEST_NO_EMOJIS_DATA_PATH="$PREPROCESSED_DATA_DIR/test_no_emojis.csv"


mkdir $PREPROCESSED_DATA_DIR
echo "Created dir $PREPROCESSED_DATA_DIR"

echo

echo "Preprocessing $TRAIN_DATA_PATH"
# The first awk call filters out those examples that do not contain [#TRIGGERWORD#]
# The second one replaces keywords with a more standard notation and adds spaces
# around them to make the job easier for the tokenizer
cat $TRAIN_DATA_PATH | awk '{if ( $0 ~ /#TRIGGERWORD#/ ) {print $0}}' \
                     | awk '{$1=""; gsub("\\[#TRIGGERWORD#\\]", " __TRIGGERWORD__ ", $0); 
                                  gsub("@USERNAME", " __USERNAME__ ", $0); 
                                  gsub("\\[NEWLINE\\]", " __NEWLINE__ ", $0); 
                                  gsub("http://url.removed", " __URL__ ", $0); print $0}' \
                     | ./utils/twokenize.py > $CLEAN_TRAIN_DATA_PATH

cat $TRAIN_DATA_PATH | awk '{if ( $0 ~ /#TRIGGERWORD#/ ) {print $1}}' > $TRAIN_LABELS_PATH

echo "Created $CLEAN_TRAIN_DATA_PATH and $TRAIN_LABELS_PATH"

echo "Getting POS tags for $TRAIN_DATA_PATH with TwiboParser"

# this script will generate the file $CLEAN_TRAIN_DATA_PATH.tagged
./utils/run_postagger.sh $CLEAN_TRAIN_DATA_PATH

# this script will generate two files:
# - $CLEAN_TRAIN_DATA_PATH.tagged.tokens
# - $CLEAN_TRAIN_DATA_PATH.tagged.pos
./utils/parse_twibo_output.py $CLEAN_TRAIN_DATA_PATH.tagged

echo


echo "Preprocessing $DEV_DATA_PATH"
# Then replace them with the real ones
paste $DEV_LABELS_PATH_ORIG $DEV_DATA_PATH > tmp.csv

# Then we repeat the same process we did for the train data
cat tmp.csv | awk '{if ( $0 ~ /#TRIGGERWORD#/ ) {print $0}}' \
            | awk '{$1="";$2=""; gsub("\\[#TRIGGERWORD#\\]", " __TRIGGERWORD__ ", $0); 
                                 gsub("@USERNAME", " __USERNAME__ ", $0); 
                                 gsub("\\[NEWLINE\\]", " __NEWLINE__ ", $0); 
                                 gsub("http://url.removed", " __URL__ ", $0); print $0}' \
            | ./utils/twokenize.py > $CLEAN_DEV_DATA_PATH

cat tmp.csv | awk '{if ( $0 ~ /#TRIGGERWORD#/ ) {print $1}}' > $DEV_LABELS_PATH
rm tmp.csv

echo "Created $CLEAN_DEV_DATA_PATH and $DEV_LABELS_PATH"
echo
echo "Getting POS tags for $CLEAN_DEV_DATA_PATH with TwiboParser"

# this script will generate the file $CLEAN_DEV_DATA_PATH.tagged
./utils/run_postagger.sh $CLEAN_DEV_DATA_PATH

# this script will generate two files:
# - $CLEAN_DEV_DATA_PATH.tagged.tokens
# - $CLEAN_DEV_DATA_PATH.tagged.pos
python ./utils/parse_twibo_output.py $CLEAN_DEV_DATA_PATH.tagged

echo
echo "Preprocessing $TEST_DATA_PATH"

cat $TEST_DATA_PATH | awk '{if ( $0 ~ /#TRIGGERWORD#/ ) {print $0}}' \
                    | awk '{$1=""; gsub("\\[#TRIGGERWORD#\\]", " __TRIGGERWORD__ ", $0); 
                                   gsub("@USERNAME", " __USERNAME__ ", $0); 
                                   gsub("\\[NEWLINE\\]", " __NEWLINE__ ", $0); 
                                   gsub("http://url.removed", " __URL__ ", $0); print $0}' \
                    | ./utils/twokenize.py > $CLEAN_TEST_DATA_PATH

cat $TEST_DATA_PATH | awk '{if ( $0 ~ /#TRIGGERWORD#/ ) {print $1}}' > $TEST_LABELS_PATH
echo "Created $CLEAN_TEST_DATA_PATH and $TEST_LABELS_PATH"
echo
# this script will generate the file $CLEAN_TRAIN_DATA_PATH.tagged
./utils/run_postagger.sh $CLEAN_TEST_DATA_PATH

# this script will generate two files:
# - $CLEAN_TEST_DATA_PATH.tagged.tokens
# - $CLEAN_TEST_DATA_PATH.tagged.pos
./utils/parse_twibo_output.py $CLEAN_TEST_DATA_PATH.tagged
echo

echo "Removing emojis from train"
cat $CLEAN_TRAIN_DATA_PATH | ./utils/remove_emojis.py > $CLEAN_TRAIN_NO_EMOJIS_DATA_PATH
echo "Created $CLEAN_TRAIN_NO_EMOJIS_DATA_PATH"

echo

echo "Removing emojis from dev"
cat $CLEAN_DEV_DATA_PATH | ./utils/remove_emojis.py > $CLEAN_DEV_NO_EMOJIS_DATA_PATH
echo "Created $CLEAN_DEV_NO_EMOJIS_DATA_PATH"

echo

echo "Removing emojis from test"
cat $CLEAN_TEST_DATA_PATH | ./utils/remove_emojis.py > $CLEAN_TEST_NO_EMOJIS_DATA_PATH
echo "Created $CLEAN_TEST_NO_EMOJIS_DATA_PATH"

