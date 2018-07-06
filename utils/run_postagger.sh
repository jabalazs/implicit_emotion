#!/usr/bin/env bash
#
# Copyright (c) 2013-2014 Lingpeng Kong
# All Rights Reserved.
#
# This file is part of TweeboParser 1.0.
#
# TweeboParser 1.0 is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# TweeboParser 1.0 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with TweeboParser 1.0.  If not, see <http://www.gnu.org/licenses/>.


CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR=$CURRENT_DIR/ark-tweet-nlp
PRETRAINED_MODEL_NAME=tagging_model

# To run the parser:

if [ "$#" -ne 1 ]; then
    echo "Usage: ./run.sh [path_to_raw_input_file_one_sentence_a_line]"
else

# Starting point:
# -- Raw text tweets, one line per tweet.
INPUT_FILE=$1

echo "Obtaining POS tags"
set -eu
java -XX:ParallelGCThreads=2 -Xmx2048m -jar $ROOT_DIR/ark-tweet-nlp-0.3.2.jar\
    --just-pos-tag\
    --model "$ROOT_DIR/$PRETRAINED_MODEL_NAME" --output-format conll ${INPUT_FILE}\
    > tagger_output.txt

# Write only 1st and 2nd columns when line is different to newline, otherwise just print a newline
OUTPUT_FILE="$1.tagged.pre"
cat tagger_output.txt | awk '{if ($0 !~ /^$/) print $1" "$2; else print;}' > $OUTPUT_FILE
echo "Created $OUTPUT_FILE."

rm tagger_output.txt


fi

