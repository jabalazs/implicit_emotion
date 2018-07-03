#!/usr/bin/python
# -*- coding: utf-8 -*-


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('parsed_file')
args = parser.parse_args()

with open(args.parsed_file, "r") as f:
    lines = f.readlines()

tagged_sentences = []
tokens = []
pos_tags = []

for line in lines:
    if line.startswith(' \n'):
        tagged_sentences.append((tokens, pos_tags))
        tokens = []
        pos_tags = []
        continue

    line = line.strip()
    if line:
        token, pos_tag = line.split()
        tokens.append(token)
        pos_tags.append(pos_tag)

tokens_file_path = args.parsed_file + '.tokens'
pos_file_path = args.parsed_file + '.pos'

with open(tokens_file_path, 'w') as tokens_file:
    with  open(pos_file_path, 'w') as pos_file:
        for tokens, pos_tags in tagged_sentences:
            txt_tokens = ' '.join(tokens)
            txt_pos =  ' '.join(pos_tags)
            tokens_file.write('{}\n'.format(txt_tokens))
            pos_file.write('{}\n'.format(txt_pos))

