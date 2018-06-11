import numpy as np
import torch
import os
import gzip
from logging import getLogger

def load_translation_data(lg1, lg2, src_file, tgt_file, n_sents = 29000, lower = True):
    assert os.path.isfile(src_file)
    assert os.path.isfile(tgt_file)
    #
    #load sentences
    data = {lg1:[], lg2:[]}
    with gzip.open(src_file) as sfile, gzip.open(tgt_file) as tfile:
        for i, (sline, tline) in enumerate(zip(sfile, tfile)):
            if i>= n_sents:
                break
            else:
                if lower:
                    sline = sline.lower()
                    tline = tline.lower()
                data[lg1].append(sline.rstrip().split())
                data[lg2].append(tline.rstrip().split())
        assert  len(data[lg1]) == len(data[lg2])

    # shuffle sentences
    return data

def sentence_evaluator(data):
    return

def load_vocabulary(lg1, src_vocabulary_file, lg2, tgt_vocabulary_file, lower = True):
    assert os.path.isfile(src_vocabulary_file)
    assert os.path.isfile(tgt_vocabulary_file)
    src_vocab, tgt_vocab = set(), set()
    with open(src_vocabulary_file) as f:
        for line in f:
            src_vocab.add(line.rstrip())

    with open(tgt_vocabulary_file) as f:
        for line in f:
            tgt_vocab.add(line.rstrip())
    vocab = {lg1:src_vocab, lg2: tgt_vocab}
    return vocab



