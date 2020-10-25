#!/usr/bin/env python3
# coding: utf-8

import sys
from sklearn.metrics import balanced_accuracy_score, accuracy_score

def without_newlines(iterable):
  return map(lambda line: line[:-1], iterable)

def main(test_labels_path, predicted_labels_path):
    with open(test_labels_path, 'r') as testL_f, \
         open(predicted_labels_path, 'r') as predL_f:
        test_labels = list(without_newlines(testL_f))
        pred_labels = list(without_newlines(predL_f))

    acc_weighted_percent = 100*balanced_accuracy_score(test_labels, pred_labels)
    acc_percent = 100*accuracy_score(test_labels, pred_labels)
    print('Accuracy (weighted, %):', round(acc_weighted_percent, 3))
    print('Accuracy (unweighted, %):', round(acc_percent, 3))

if __name__ == '__main__':
    if len(sys.argv) != 2+1:
        print(f'Usage: {sys.argv[0]} <test labels path> <predicted labels path>')
        sys.exit(1)
    main(*sys.argv[1:])
