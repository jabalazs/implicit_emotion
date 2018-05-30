#!/bin/bash

cat data/train.csv | awk '{$1=""; gsub("\\[#TRIGGERWORD#\\]", "__TRIGGERWORD__", $0); 
                                  gsub("@USERNAME", "__USERNAME__", $0); 
                                  gsub("\\[NEWLINE\\]", "__NEWLINE__", $0); 
                                  gsub("http://url.removed", "__URL__", $0); print $0}' \
                   | ./twokenize.py > data/train_clean.csv

cat data/train.csv | awk '{print $1}' > data/train_labels.csv
