#!/usr/bin/python

import argparse
import csv
import numpy as np

from src.config import LABEL2ID

parser = argparse.ArgumentParser(
                   description='Obtain predicted classes by averaging '
                               'several softmax outputs.')

parser.add_argument('files', type=str, nargs='+',
                    help='prob files from which to generate the predictions')


def get_ids_and_ndarray_from_prob_file(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)
        numeric_probs = [[float(elem) for elem in str_probs] for str_probs in lines]
    return np.array(numeric_probs)


def run_main():
    args = parser.parse_args()
    print(args)
    filenames = args.files
    array_list = []

    for filename in filenames:
        probs_array = get_ids_and_ndarray_from_prob_file(filename)
        array_list.append(probs_array)

    tensor = np.stack(array_list, axis=2)
    mean_probs = np.mean(tensor, axis=2)

    label_ids = np.argmax(mean_probs, axis=1).tolist()
    id2label = {v: k for k, v in LABEL2ID.items()}
    labels = [id2label[label_id] for label_id in label_ids]

    preds_filename = 'ensembled_predictions.txt'
    print('Writing {}'.format(preds_filename))
    with open(preds_filename, 'w') as f:
        for pred_label in labels:
            f.write(pred_label + '\n')


if __name__ == "__main__":
    run_main()
