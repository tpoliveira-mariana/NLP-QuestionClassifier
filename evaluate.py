#!/usr/bin/env python3
# coding: utf-8

import sys
from sklearn.metrics import balanced_accuracy_score, accuracy_score

RESET='\u001b[0m'
RED='\u001b[31m'
GREEN='\u001b[32m'

def without_newlines(iterable):
    return map(lambda line: line[:-1], iterable)

def read_labels(path):
    with open(path, 'r') as f:
       return list(without_newlines(f))

def format_diff(new, old):
    diff = new - old

    color = ''
    prefix = ' '
    if diff > 0:
        color = GREEN
        prefix='+'
    elif diff < 0:
        color = RED
        prefix='-'

    return f'{color}({prefix}{round(diff, 3)}){RESET}'

def calc_acc_percent(test_labels, pred_labels):
    acc_weighted_percent = 100*balanced_accuracy_score(test_labels, pred_labels)
    acc_percent = 100*accuracy_score(test_labels, pred_labels)

    return acc_percent, acc_weighted_percent

def main(test_labels_path, pred_labels_path, old_pred_path=None):
    test_labels = read_labels(test_labels_path)
    pred_labels = read_labels(pred_labels_path)
    old_pred_labels = None if old_pred_path is None else read_labels(old_pred_path)


    acc, acc_weighted = calc_acc_percent(test_labels, pred_labels)

    if old_pred_labels is None:
        print('Accuracy (weighted, %):', round(acc_weighted, 3))
        print('Accuracy (unweighted, %):', round(acc, 3))
    else:
        old_acc, old_acc_weighted = calc_acc_percent(test_labels, old_pred_labels)
        acc_diff = format_diff(acc, old_acc)
        acc_weighted_diff = format_diff(acc_weighted, old_acc_weighted)

        print('Accuracy (weighted, %):', round(acc_weighted, 3), acc_weighted_diff)
        print('Accuracy (unweighted, %):', round(acc, 3), acc_diff)

        if acc < old_acc and acc_weighted < old_acc_weighted:
            print('you got worse :(')
            sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) != 2+1 and len(sys.argv) != 3+1:
        print(f'Usage: {sys.argv[0]} <test labels path> <predicted labels path> [<old predicted labels path for comparison>]')
        sys.exit(1)
    main(*sys.argv[1:])
