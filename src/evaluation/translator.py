# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from copy import deepcopy
import numpy as np
import torch
from torch.autograd import Variable
import os
import kenlm

from . import get_wordsim_scores, get_crosslingual_wordsim_scores
from . import get_word_translation_accuracy
from . import load_europarl_data, get_sent_translation_accuracy
from ..dico_builder import get_candidates, build_dictionary
from ..utils import get_nn_avg_dist
from math import exp
from math import log
logger = getLogger()



class Translator(object):
    def __init__(self, trainer):
        self.emb1 = trainer.src_emb.weight.data
        self.emb2 = trainer.tgt_emb.weight.data
        self.src_dico = trainer.src_dico
        self.tgt_dico = trainer.tgt_dico
        self.mapping = trainer.mapping
        self.params = trainer.params

        self.emb1 = self.emb1 / self.emb1.norm(2, 1, keepdim=True).expand_as(self.emb1)
        self.emb2 = self.emb2 / self.emb2.norm(2, 1, keepdim=True).expand_as(self.emb2)
        self.word_vec1 = dict([(w, self.emb1[self.src_dico.word2id[w]]) for w in self.src_dico.word2id])
        self.word_vec2 = dict([(w, self.emb2[self.tgt_dico.word2id[w]]) for w in self.tgt_dico.word2id])

        ###########################################
        self.average_dist1 = get_nn_avg_dist(self.emb2, self.emb1, 10)
        self.average_dist2 = get_nn_avg_dist(self.emb1, self.emb2, 10)
        self.average_dist1 = torch.from_numpy(self.average_dist1).type_as(self.emb1)
        self.average_dist2 = torch.from_numpy(self.average_dist2).type_as(self.emb2)






    def word_translation(self, to_log, word, lg1, lg2, method, topk=100,vocab=None, keep=False):

        if word not in self.src_dico.word2id:
            ### keep the origin word without translation or use <unk> to mark it
            if keep:
                return [word], [0.01]
            else:
                return ['<unk>'],[0.01]

        elif word == "$number" or word == "$url":
            return [word], [0.01]
        else:
            word_vec= self.word_vec1[word].view(1,300)

        if method == 'knn':
            scores = word_vec.mm(self.emb2.transpose(0,1))
        elif method =='csls':
            scores = word_vec.mm(self.emb2.transpose(0,1))
            scores.mul_(2)
            idx = self.src_dico.word2id[word]
            scores.sub_(self.average_dist1[[idx]][:, None] + self.average_dist2[None, :])

        scores = scores.topk(topk, 1, True)
        top_matches = scores[1]
        prob = scores[0].cpu().numpy()[0]

        prob = ((prob+1)/2).tolist()
        word_translation, word_probability = [], []
        if vocab:
            for i in top_matches:
                for j in i:
                    word = self.tgt_dico.id2word[j]
                    if word in vocab[lg2]:
                        word_translation.append(word)
        else:
            for i in top_matches:
                for j in i:
                    word = self.tgt_dico.id2word[j]
                    if word != "$number" and word != "$url" and word != "</s>":
                        word_translation.append(word)


        if len(word_translation)==0:
            word_translation=[word]
            prob=[0]

        return word_translation, prob



    def sent_translation_w(self, sent, lg1, lg2, to_log, method, topk=100, vocab=None, keep=True):

        tran_sent = []
        for word in sent:
            if type(word)==bytes:
                word = word.decode()
            words,_ = self.word_translation(to_log, word.lower(),lg1, lg2, method, topk, vocab, keep=keep)
            word=words[0]
            tran_sent.append(word)
        return tran_sent

    # def sent_translation_lm(self, sent, lm, lg1, lg2, to_log, method, topk=5, beam_size=10, vocab=None,lm_scaling=0, lex_scaling=1, keep=False):
    #     t_max =0
    #     t_min = 1
    #     sequences = [[list(), 0.0]]
    #     for word_idx in range(len(sent)):
    #         src_word = sent[word_idx]
    #         all_candidates = list()
    #         if type(src_word)==bytes:
    #             src_word = src_word.decode()
    #         lex_words,lex_probs = self.word_translation(to_log, src_word.lower() ,lg1, lg2, method, topk, vocab, keep)#to_log, word, vocab, lg1, lg2, method, k=5
    #
    #         if lex_words:
    #             word_candidates = list(zip(lex_words, lex_probs))
    #             for [seq, score] in sequences:
    #                 pre_prob = lm.score(' '.join(seq), bos=True, eos=False)
    #                 for word, prob in word_candidates:
    #                     if word_idx<len(sent)-1:
    #                         candidate = [seq+[word], score-(lex_scaling*log(prob)+lm_scaling*(lm.score(' '.join(seq+[word]), bos=True, eos=False)-pre_prob))]
    #                     else:
    #                         candidate = [seq+[word], score-(lex_scaling*log(prob)+lm_scaling*(lm.score(' '.join(seq+[word]), bos=True, eos=True)-pre_prob))]
    #                     all_candidates.append(candidate)
    #         elif not lex_words:
    #             for seq, score in sequences:
    #                 word_candidates = self.lm_predict(seq, lm)
    #                 for word, prob in word_candidates:
    #                     candidate = [seq+[word], score-prob]
    #                     all_candidates.append(candidate)
    #         ordered = sorted(all_candidates, key=lambda tup:tup[1])
    #         sequences = ordered[:beam_size]
    #
    #     return sequences[0][0], t_max, t_min

    def sent_translation_lm(self, sent, lm, lg1, lg2, to_log, method, topk=5, beam_size=10, vocab=None,lm_scaling=0, lex_scaling=1, keep=False):
        t_max =0
        t_min = 1
        sequences = [[list(), 0.0]]
        for word_idx in range(len(sent)):
            src_word = sent[word_idx]
            all_candidates = list()
            if type(src_word)==bytes:
                src_word = src_word.decode()
            lex_words,lex_probs = self.word_translation(to_log, src_word.lower() ,lg1, lg2, method, topk, vocab, keep)#to_log, word, vocab, lg1, lg2, method, k=5

            if lex_words:
                word_candidates = list(zip(lex_words, lex_probs))
                for [seq, score] in sequences:
                    pre_prob = lm.score(' '.join(seq), bos=True, eos=False)
                    for word, prob in word_candidates:
                        if word_idx<len(sent)-1:
                            candidate = [seq+[word], score-(lex_scaling*log(prob)+lm_scaling*(lm.score(' '.join(seq+[word]), bos=True, eos=False)-pre_prob))]
                        else:
                            candidate = [seq+[word], score-(lex_scaling*log(prob)+lm_scaling*(lm.score(' '.join(seq+[word]), bos=True, eos=True)-pre_prob))]
                        all_candidates.append(candidate)
            elif not lex_words:
                for seq, score in sequences:
                    word_candidates = self.lm_predict(seq, lm)
                    for word, prob in word_candidates:
                        candidate = [seq+[word], score-prob]
                        all_candidates.append(candidate)
            ordered = sorted(all_candidates, key=lambda tup:tup[1])
            sequences = ordered[:beam_size]

        return sequences[0][0], t_max, t_min

    def corpus_translation_w(self, data, lg1, lg2, to_log, method, topk=100, beam_size=10, vocab=None, lm=None, lex_scaling=1, lm_scaling=1, keep=False, T=1):
        cnt = 0
        translation = []
        t_max = 0
        t_min=1
        for sent in data[lg1]:
            cnt+=1
            logger.info(cnt)
            if lm!="":
                tran_sent, max, min = self.sent_translation_lm(sent, lm, lg1, lg2, to_log, method, topk=topk, beam_size=beam_size, lex_scaling=lex_scaling, lm_scaling=lm_scaling, keep=keep)
                if max>t_max:
                    t_max =  max
                if min< t_min:
                    t_min= min
            else:
                tran_sent = self.sent_translation_w(sent, lg1, lg2, to_log, method, beam_size, vocab)
            translation.append(tran_sent)

        translation_path = os.path.join(self.params.exp_path, 'sent-translation.%s' % self.params.tgt_lang)
        with open(translation_path, 'w') as f:
            for item in translation:
                f.write("%s\n" % ' '.join(item))
        logger.info('Writing translation results to %s ...' % translation_path)

    def lm_predict(self, seq, lm, topk=100):
        lm_candidates = list()
        pre_score = lm.score(' '.join(seq), bos=True, eos=False)
        for word in self.tgt_dico.word2id:
            lm_candidates.append([word, lm.score(' '.join(seq+[word]), bos=True, eos=False)- pre_score])
        predicted = sorted(lm_candidates, key=lambda x:x[1], reverse=True)[:100]
        return predicted







