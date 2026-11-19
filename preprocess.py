#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Binarize a (BPE-segmented) text file with a vocabulary.

    python preprocess.py <vocab.txt> <data.txt>

vocab.txt : one "<token> <count>" pair per line (e.g. the output of fastBPE `getvocab`)
data.txt  : one sentence per line, tokens separated by spaces
Output    : <data.txt>.pth, next to the input file. An already existing .pth is re-used, delete it to
            re-binarize.

The training scripts expect the following layout inside --data_path
(binarize every text file once, the names below are the *text* files):
    train.<lang>   valid.<lang>   test.<lang>                      monolingual data
    valid.<l1>-<l2>.<l1>  valid.<l1>-<l2>.<l2>                      parallel data used for evaluation
    test.<l1>-<l2>.<l1>   test.<l1>-<l2>.<l2>                       (l1 < l2 alphabetically, same number of lines)
"""

import os
import argparse

from xlm.logger import create_logger
from xlm.data.dictionary import Dictionary


def main(voc_path, txt_path):
    logger = create_logger(None, 0)
    assert os.path.isfile(voc_path), voc_path
    assert os.path.isfile(txt_path), txt_path
    bin_path = txt_path + '.pth'

    dico = Dictionary.read_vocab(voc_path)
    logger.info("")

    data = Dictionary.index_data(txt_path, bin_path, dico)
    logger.info("%i words (%i unique) in %i sentences." % (
        len(data['sentences']) - len(data['positions']),
        len(data['dico']),
        len(data['positions'])
    ))
    if len(data['unk_words']) > 0:
        logger.info("%i unknown words (%i unique), covering %.2f%% of the data." % (
            sum(data['unk_words'].values()),
            len(data['unk_words']),
            sum(data['unk_words'].values()) * 100. / (len(data['sentences']) - len(data['positions']))
        ))
        if len(data['unk_words']) < 30:
            for w, c in sorted(data['unk_words'].items(), key=lambda x: x[1])[::-1]:
                logger.info("%s: %i" % (w, c))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Binarize a text file with a vocabulary (see module docstring).")
    parser.add_argument("vocab", help="vocabulary file, one '<token> <count>' per line")
    parser.add_argument("text", help="text file to binarize; <text>.pth is written")
    args = parser.parse_args()
    main(args.vocab, args.text)
