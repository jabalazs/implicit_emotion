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

# Get the path of the components of TweeboParser
#ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

ROOT_DIR=~/TweeboParser
SCRIPT_DIR="${ROOT_DIR}/scripts"
TAGGER_DIR="${ROOT_DIR}/ark-tweet-nlp-0.3.2"
PARSER_DIR="${ROOT_DIR}/TBParser"
TOKENSEL_DIR="${ROOT_DIR}/token_selection"
MODEL_DIR="${ROOT_DIR}/pretrained_models"
WORKING_DIR="${ROOT_DIR}/working_dir"

# To run the parser:

if [ "$#" -ne 1 ]; then
    echo "Usage: ./run.sh [path_to_raw_input_file_one_sentence_a_line]"
else

mkdir -p $WORKING_DIR

# Starting point:
# -- Raw text tweets, one line per tweet.
INPUT_FILE=$1

echo "Obtaining POS tags"
set -eu
java -XX:ParallelGCThreads=2 -Xmx2048m -jar $TAGGER_DIR/ark-tweet-nlp-0.3.2.jar\
    --model ${MODEL_DIR}/tagging_model --output-format conll ${INPUT_FILE}\
    > ${WORKING_DIR}/tagger_output.txt

# Write only 1st and 2nd columns when line is different to newline, otherwise just print a newline
OUTPUT_FILE="$1.tagged"
cat ${WORKING_DIR}/tagger_output.txt | awk '{if ($0 !~ /^$/) print $1" "$2; else print;}' > $OUTPUT_FILE
echo "Created $OUTPUT_FILE."

fi

