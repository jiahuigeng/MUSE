import numpy as np
import torch
import os
import gzip
from logging import getLogger
import os
from logging import getLogger
import io
logger = getLogger()
def load_bi_translation_data(lg1, lg2, src_file, tgt_file, n_sents = 29000, lower = True):
    assert os.path.isfile(src_file)
    assert os.path.isfile(tgt_file)
    #
    #load sentences
    data = {lg1:[], lg2:[]}
    if '.gz' not in src_file:
        with open(src_file) as sfile, open(tgt_file) as tfile:
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

    else:
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

def load_translation_data(filename, n_sent=1000000000, lower=True):
    res_data = []
    if filename.endswith(".gz"):
        data = gzip.open(filename)
    else:
        data = open(filename)
    for i, sent in enumerate(data):
        if i>n_sent:
            break
        else:
            if lower:
                sent = sent.lower()
            res_data.append(sent.rstrip().split())
    return res_data

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


def read_txt_embeddings(params, source, full_vocab=False):
    """
    Reload pretrained embeddings from a text file.
    """
    word2id = {}
    vectors = []
    # load pretrained embeddings
    #src_max_vocab = params.src_max_vocab if params.src_max_vocab!=1 else params.max_vocab
    #tgt_max_vocab = params.tgt_max_vocab if params.tgt_max_vocab!=1 else params.max_vocab

    if params.src_range!="":
        src_start, src_end = map(int, params.src_range.split("_"))
        src_max_vocab = src_end - src_start + 1
    else:
        src_start, src_end = 1, params.max_vocab
        src_max_vocab = params.max_vocab

    if params.tgt_range!="":
        tgt_start, tgt_end = map(int, params.tgt_range.split("_"))
        tgt_max_vocab =  tgt_end - tgt_start + 1
    else:
        tgt_start, tgt_end = 1, params.max_vocab
        tgt_max_vocab = params.max_vocab

    lang = params.src_lang if source else params.tgt_lang
    emb_path = params.src_emb if source else params.tgt_emb
    _emb_dim_file = params.emb_dim
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            if i == 0:
                split = line.split()
                assert len(split) == 2
                assert _emb_dim_file == int(split[1])
            else:
                if source and i in range(src_start, src_end+1):
                    word, vect = line.rstrip().split(' ', 1)
                    if not full_vocab:
                        word = word.lower()
                    vect = np.fromstring(vect, sep=' ')
                    if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                        vect[0] = 0.01
                    if word in word2id:
                        if full_vocab:
                            logger.warning("Word '%s' found twice in %s embedding file"
                                           % (word, 'source' if source else 'target'))
                    else:
                        if not vect.shape == (_emb_dim_file,):
                            logger.warning("Invalid dimension (%i) for %s word '%s' in line %i."
                                           % (vect.shape[0], 'source' if source else 'target', word, i))
                            continue
                        assert vect.shape == (_emb_dim_file,), i
                        word2id[word] = len(word2id)
                        vectors.append(vect[None])
                elif not source and i in range(tgt_start, tgt_end+1):
                    word, vect = line.rstrip().split(' ', 1)
                    if not full_vocab:
                        word = word.lower()
                    vect = np.fromstring(vect, sep=' ')
                    if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                        vect[0] = 0.01
                    if word in word2id:
                        if full_vocab:
                            logger.warning("Word '%s' found twice in %s embedding file"
                                           % (word, 'source' if source else 'target'))
                    else:
                        if not vect.shape == (_emb_dim_file,):
                            logger.warning("Invalid dimension (%i) for %s word '%s' in line %i."
                                           % (vect.shape[0], 'source' if source else 'target', word, i))
                            continue
                        assert vect.shape == (_emb_dim_file,), i
                        word2id[word] = len(word2id)
                        vectors.append(vect[None])
                if source and len(word2id) > src_max_vocab:
                    break
                elif not source and len(word2id) > tgt_max_vocab:
                    break

    assert len(word2id) == len(vectors)
    logger.info("Loaded %i pre-trained word embeddings." % len(vectors))

    # compute new vocabulary / embeddings
    id2word = {v: k for k, v in word2id.items()}
    dico = Dictionary(id2word, word2id, lang)
    embeddings = np.concatenate(vectors, 0)
    embeddings = torch.from_numpy(embeddings).float()
    embeddings = embeddings.cuda() if (params.cuda and not full_vocab) else embeddings

    assert embeddings.size() == (len(dico), params.emb_dim)
    return dico, embeddings
