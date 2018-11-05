# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from logging import getLogger
import scipy
import scipy.linalg
import torch
import numpy as np
import string
import re
import random
from random import shuffle
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import Sigmoid
from .utils import get_optimizer, load_embeddings, normalize_embeddings, export_embeddings
from .utils import clip_parameters
from .dico_builder import build_dictionary, build_w_dictionary
from .evaluation.word_translation import DIC_EVAL_PATH, load_identical_char_dico, load_dictionary
from torch.nn import MSELoss

logger = getLogger()


class Trainer(object):

    def __init__(self, src_emb, tgt_emb, mapping, discriminator, params):
        """
        Initialize trainer script.
        """
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.src_dico = params.src_dico
        self.tgt_dico = getattr(params, 'tgt_dico', None)
        self.mapping = mapping
        self.discriminator = discriminator
        self.params = params
        self.index = 0
        logger.info(params.init_lr)
        logger.info("the trainer emb information")
        logger.info(self.src_emb.weight.data[0][0])
        self.m_optimizer = torch.optim.Adam(self.mapping.parameters(), lr=params.init_lr)
        self.tt =  Variable(torch.ones(self.params.batch_size)).cuda() if self.params.cuda else Variable(torch.ones(self.params.batch_size))
        # optimizers

        if hasattr(params, 'map_optimizer'):
            optim_fn, optim_params = get_optimizer(params.map_optimizer)
            self.map_optimizer = optim_fn(mapping.parameters(), **optim_params)
        if hasattr(params, 'dis_optimizer'):
            optim_fn, optim_params = get_optimizer(params.dis_optimizer)
            self.dis_optimizer = optim_fn(discriminator.parameters(), **optim_params)
        else:
            assert discriminator is None

        # best validation score
        self.best_valid_metric = -1e12

        self.decrease_lr = False

    def get_dis_xy(self, volatile):
        """
        Get discriminator input batch / output target.
        """
        # select random word IDs
        bs = self.params.batch_size
        mf = self.params.dis_most_frequent
        assert mf <= min(len(self.src_dico), len(self.tgt_dico))
        src_ids = torch.LongTensor(bs).random_(len(self.src_dico) if mf == 0 else mf)
        tgt_ids = torch.LongTensor(bs).random_(len(self.tgt_dico) if mf == 0 else mf)
        if self.params.cuda:
            src_ids = src_ids.cuda()
            tgt_ids = tgt_ids.cuda()

        # get word embeddings
        src_emb = self.src_emb(Variable(src_ids, volatile=True))
        tgt_emb = self.tgt_emb(Variable(tgt_ids, volatile=True))
        logger.info("before mapping")
        logger.info(src_emb)
        src_emb = self.mapping(Variable(src_emb.data, volatile=volatile))
        logger.info("after mapping")
        logger.info(src_emb)
        tgt_emb = Variable(tgt_emb.data, volatile=volatile)

        # input / target
        x = torch.cat([src_emb, tgt_emb], 0)
        y = torch.FloatTensor(2 * bs).zero_()
        y[:bs] = 1 - self.params.dis_smooth
        y[bs:] = self.params.dis_smooth
        y = Variable(y.cuda() if self.params.cuda else y)

        return x, y

    def dis_step(self, stats):
        """
        Train the discriminator.
        """
        self.discriminator.train()

        # loss
        x, y = self.get_dis_xy(volatile=True)
        preds = self.discriminator(Variable(x.data))
        loss = F.binary_cross_entropy(preds, y)
        stats['DIS_COSTS'].append(loss.data[0])

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected (discriminator)")
            exit()

        # optim
        self.dis_optimizer.zero_grad()
        loss.backward()
        self.dis_optimizer.step()
        clip_parameters(self.discriminator, self.params.dis_clip_weights)


    def mapping_step(self, stats):
        """
        Fooling discriminator training step.
        """
        if self.params.dis_lambda == 0:
            return 0

        self.discriminator.eval()

        # loss
        x, y = self.get_dis_xy(volatile=False)
        preds = self.discriminator(x)
        loss = F.binary_cross_entropy(preds, 1 - y)
        loss = self.params.dis_lambda * loss

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected (fool discriminator)")
            exit()

        # optim
        self.map_optimizer.zero_grad()
        loss.backward()
        self.map_optimizer.step()
        self.orthogonalize()

        return 2 * self.params.batch_size

    def load_training_dico(self, dico_train):
        """
        Load training dictionary.
        """
        word2id1 = self.src_dico.word2id
        word2id2 = self.tgt_dico.word2id

        # identical character strings
        if dico_train == "identical_char":
            self.dico = load_identical_char_dico(word2id1, word2id2)
        # use one of the provided dictionary
        elif dico_train == "default":
            filename = '%s-%s.0-5000.txt' % (self.params.src_lang, self.params.tgt_lang)
            self.dico = load_dictionary(
                os.path.join(DIC_EVAL_PATH, filename),
                word2id1, word2id2
            )
        # dictionary provided by the user
        else:
            self.dico = load_dictionary(dico_train, word2id1, word2id2)

        # cuda
        if self.params.cuda:
            self.dico = self.dico.cuda()
            #self.fix_dico = self.dico.cuda()

    def build_dictionary(self):
        """
        Build a dictionary from aligned embeddings.
        """
        src_emb = self.mapping(self.src_emb.weight).data
        tgt_emb = self.tgt_emb.weight.data
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)

        id2word1 = self.src_dico.id2word
        id2word2 = self.tgt_dico.id2word
        self.dico =  build_dictionary(src_emb, tgt_emb, self.params)
        # logger.info("dico size: before")
        # logger.info(self.dico.shape)

        #self.dico = torch.cat([self.dico, build_dictionary(src_emb, tgt_emb, self.params)])
        '''
        if self.params.cuda:
            unique_list = list(set([(a, b) for [a, b] in self.dico.cpu().numpy()]))
        else:
            unique_list = list(set([(a, b) for [a, b] in self.dico.numpy()]))
        
        if self.params.cuda:
            self.dico = self.dico.cuda()

        # if self.params.cuda and self.params.save_dico:
        #     for [a,b] in self.dico.cpu().numpy():
        #         print(id2word1[a], id2word2[b])
        # elif not self.params.cuda and self.params.sava_dico:
        #     for [a,b] in self.dico.numpy():
        #         print(id2word1[a], id2word2[b])
             perp.append(model.perplexity(sent))
        '''

        # if self.params.cuda:
        #     for [a,b] in self.dico.cpu().numpy():
        #         print(id2word1[a], id2word2[b])
        # elif not self.params.cuda:
        #     for [a,b] in self.dico.numpy():
        #         print(id2word1[a], id2word2[b])

        logger.info("dico_size: after")
        logger.info(self.dico.shape)

    def build_w_dictionary(self):
        src_emb = self.mapping(self.src_emb.weight).data
        tgt_emb = self.tgt_emb.weight.data
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
        self.dico =  build_w_dictionary(src_emb, tgt_emb, self.params)
        logger.info("dico_size: after")
        logger.info(self.dico.shape)

    def get_lm_xy(self, volatile, total=False):
        bs = self.params.batch_size
        if self.index < len(self.dico):
            ids = torch.LongTensor(range(self.index, min(len(self.dico), self.index+bs)))
            self.index+=bs
        if self.index >= len(self.dico):
            self.index = 0
        # ids = torch.LongTensor(bs).random_(len(self.dico))
        # get word embeddings
        if self.params.cuda:
            ids = ids.cuda()
        src_emb = self.src_emb.weight.data[self.dico[ids][:,0]]
        tgt_emb = self.tgt_emb.weight.data[self.dico[ids][:,1]]
        if self.params.cuda:
            src_emb = src_emb.cuda()
            tgt_emb = tgt_emb.cuda()

        if total:
            src_emb = self.src_emb.weight.data[self.dico[:, 0]]
            tgt_emb = self.tgt_emb.weight.data[self.dico[:, 1]]
            if self.params.cuda:
                src_emb = src_emb.cuda()
                tgt_emb = tgt_emb.cuda()
                return src_emb, tgt_emb

        if self.params.wt_scaling != -1:
            weight = self.weight.data[ids]
            if self.params.cuda:
                weight = weight.cuda()
            return src_emb, tgt_emb, weight
        
        return src_emb, tgt_emb

    def get_train_xy(self, volatile):
        bs = self.params.batch_size

        ids = torch.LongTensor(bs).random_(len(self.dico))
        #get word embeddings
        if self.params.cuda:
            ids = ids.cuda()
        src_emb = self.src_emb.weight.data[self.dico[ids][:,0]]
        tgt_emb = self.tgt_emb.weight.data[self.dico[ids][:,1]]
        if self.params.cuda:
            src_emb = src_emb.cuda()
            tgt_emb = tgt_emb.cuda()
        return src_emb, tgt_emb
#------------------------------------------------------------------------------------------
    def ransac(self, n, k, t, d):
        """
        n: num of pairs for procrustes training
        k: num of ransac iterations
        t: threshold for inliers
        d: num of inliers to assert model fits
        """
        n_iter = 0
        bestfit = None
        besterr = np.inf
        best_inlier_points = None
        self.mapping.weight.data.copy_(torch.zeros(self.params.emb_dim, self.params.emb_dim))
        logger.info("ransac with n={}, k={}, t={}, d={}".format(n,k,t,d))
        while n_iter < k:
            train_idxs, test_idxs = self.ransac_partition(n)
            self.ransac_procrustes(train_idxs)
            #test_err = self.ransac_eval(test_idxs)
            #also_inliers = (test_err<t).cpu().nonzero().squeeze().data.numpy().tolist()
            
            all_err = self.ransac_eval(train_idxs+test_idxs)
            mean_err = torch.mean(all_err).cpu().data[0]
            all_inliers = (all_err<mean_err*t).cpu().nonzero().squeeze().data.numpy().tolist()
            if len(all_inliers)> len(self.dico)*d:
                #betteridxs = train_idxs+also_inliers
                betteridxs = all_inliers
                bettermodel = self.ransac_procrustes(betteridxs)
                bettererr = self.ransac_eval(betteridxs)
                thiserr = torch.mean(bettererr).cpu().data[0]
                if thiserr< besterr:
                    logger.info("updated:")
                    bestfit = bettermodel
                    besterr = thiserr
                    best_inlier_points = betteridxs
                    logger.info(besterr)
            n_iter+=1

        if bestfit is None:
            logger.info("model not found")
            return False
        else:
            logger.info("err info:")
            logger.info(besterr)
            logger.info(len(best_inlier_points))
            W = self.mapping.weight.data
            W.copy_(bestfit.type_as(W))
            self.orthogonalize()
            return True



    def ransac_partition(self, n):
        all_idxs = [i for i in range(len(self.dico))]
        np.random.shuffle(all_idxs)
        idxs1 = all_idxs[:n]
        idxs2 = all_idxs[n:]
        return idxs1, idxs2

    def ransac_eval(self, idxs):
        x = self.src_emb.weight.data[self.dico[idxs][:, 0]]
        y = self.tgt_emb.weight.data[self.dico[idxs][:, 1]]
        x = Variable(x)
        y = Variable(y)
        mapped_x = self.mapping(x)
        return torch.sum((mapped_x-y)**2, 1)

    def ransac_procrustes(self, idxs):
        A = self.src_emb.weight.data[self.dico[idxs][:, 0]]
        B = self.tgt_emb.weight.data[self.dico[idxs][:, 1]]
        W = self.mapping.weight.data
        M = B.transpose(0, 1).mm(A).cpu().numpy()
        U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
        W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))
        self.orthogonalize()
        return torch.from_numpy(U.dot(V_t))


    #------------------------------------------------------------------------------------------
    def weighted_procrustes(self):
            logger.info("Training with Procrustes")
            logger.info("dico size:"+str(len(self.dico)))
            A = self.src_emb.weight.data[self.dico[:, 0]]
            logger.info(A)
            logger.info(self.weight)
            Aw = A*self.weight.data
            B = self.tgt_emb.weight.data[self.dico[:, 1]]
            Bw = B*self.weight.data
            W = self.mapping.weight.data
            M = Bw.transpose(0, 1).mm(Aw).cpu().numpy()
            U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
            W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))

    #------------------------------------------------------------------------------------------
    def single_statitics(self, file):
        word_statistics = dict()
        word2id = self.src_dico.word2id
        for w in word2id:
            word_statistics[word2id[w]] = 0

        for sent in file:
            for word in sent:
                if word in word2id:
                    idx = word2id[word]
                    word_statistics[idx]+=1

        # norm = sum(word_statistics.values())
        # for idx in word_statistics:
        #     word_statistics[idx] = word_statistics[w]/norm
        return word_statistics
    
    def double_statistics(self, src_file, tgt_file):
        pair_statistics = dict()
        word2id1 = self.src_dico.word2id
        word2id2 = self.tgt_dico.word2id

    def freq_procrustes(self, src_stats, tgt_stats=None):
        logger.info("Training with Procrustes")
        logger.info("dico size:"+str(len(self.dico)))
        if not tgt_stats:
            idxs = [i for i in range(len(self.dico)) if src_stats[self.dico[i][0]]>=self.params.freq_th]
            A = self.src_emb.weight.data[self.dico[idxs][:, 0]]
            B = self.tgt_emb.weight.data[self.dico[idxs][:, 1]]
            logger.info(A.shape)
            W = self.mapping.weight.data
            M = B.transpose(0, 1).mm(A).cpu().numpy()
            U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
            W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))
        else:
            src_idxs = [i for i in range(len(self.dico)) if src_stats[self.dico[i][0]]>=self.params.freq_th]
            tgt_idxs = [i for i in range(len(self.dico)) if src_stats[self.dico[i][1]]>=self.params.freq_th]
            if self.params.freq_op == 0:
                idxs = src_idxs
            elif self.params.freq_op == 1:
                idxs = tgt_idxs
            elif self.params.freq_op == 2:
                idxs = src_idxs and tgt_idxs
            elif self.params.freq_op == 3:
                idxs = src_idxs or tgt_idxs

            A = self.src_emb.weight.data[self.dico[idxs][:, 0]]
            B = self.tgt_emb.weight.data[self.dico[idxs][:, 1]]
            logger.info(A.shape)
            W = self.mapping.weight.data
            M = B.transpose(0, 1).mm(A).cpu().numpy()
            U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
            W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))

    def freq_filter(self, src_stats, tgt_stats=None):
        logger.info("Training with Procrustes")
        logger.info("dico size:"+str(len(self.dico)))
        idxs = None
        if not tgt_stats:
            idxs = [i for i in range(len(self.dico)) if src_stats[self.dico[i][0]]>=self.params.freq_th]
            A = self.src_emb.weight.data[self.dico[idxs][:, 0]]
            B = self.tgt_emb.weight.data[self.dico[idxs][:, 1]]
            logger.info(A.shape)
            W = self.mapping.weight.data
            M = B.transpose(0, 1).mm(A).cpu().numpy()
            U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
            W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))
        else:
            src_idxs = [i for i in range(len(self.dico)) if src_stats[self.dico[i][0]]>=self.params.freq_th]
            tgt_idxs = [i for i in range(len(self.dico)) if src_stats[self.dico[i][1]]>=self.params.freq_th]
            if self.params.freq_op == 0:
                idxs = src_idxs
            elif self.params.freq_op == 1:
                idxs = tgt_idxs
            elif self.params.freq_op == 2:
                idxs = src_idxs and tgt_idxs
            elif self.params.freq_op == 3:
                idxs = src_idxs or tgt_idxs

            self.dico = self.dico[idxs]
        #---------------
        # filtering

    #------------------------------------------------------------------------------------------
    def load_lm_dictionay(self, src_data, tgt_data, samples=5000):
        word2id1 = self.src_dico.word2id
        word2id2 = self.tgt_dico.word2id

        pairs = []
        for s_sent, t_sent in zip(src_data, tgt_data):
            for s_word, t_word in zip(s_sent, t_sent):
                if not re.search(r'\d', s_word) and not re.search(r'\d', t_word) and "number" not in s_word and "number" not in t_word and s_word in word2id1 and t_word in word2id2:
                    pairs.append((s_word, t_word))
        pairs = list(pairs)#
        # ids = [elem[0] for elem in pairs]
        # s_id = set(iid for iid in ids if ids.count(iid)>1)
        # pairs = [elem for elem in pairs if elem[0] in s_id]
        #print(pairs)
        #shuffle(pairs) ##############
        logger.info("Found %i pairs of words in the dictionary (%i unique)" % (len(pairs), len(set([x for x, _ in pairs]))))
        # for item in pairs:
        #     print(word2id1[item[0]], word2id2[item[1]])

        if len(pairs)==0:
            logger.info("No pairs of word found")
        else:
            dico = torch.LongTensor(len(pairs), 2)
            for i, (word1, word2) in enumerate(pairs):
                dico[i, 0] = word2id1[word1]
                dico[i, 1] = word2id2[word2]

        self.dico = dico.cuda() if self.params.cuda else dico
        logger.info("Loaded translation dictionary")
        #return dico.cuda() if self.params.cuda else dico

    '''

    def weighted_mse_loss(self, preds, target, weights):
        diff = (preds - target)**2
        diff = torch.sum(diff, dim=1) * weights
        loss = diff.sum(0)
        return loss
    '''

    def weight_distribution(self):
        x, y = self.get_lm_xy(volatile=False, total=True)
        x = Variable(x)
        y = Variable(y)
        sigmoid = Sigmoid()
        logger.info("total wf info:")
        mapped = self.mapping(x)
        mapped = mapped / mapped.norm(2, 1, keepdim=True).expand_as(mapped)
        y = y/ y.norm(2, 1, keepdim=True).expand_as(y)
        logger.info(mapped.shape)
        logger.info("total e info:")
        logger.info(y.shape)
        weight = torch.sum(mapped*y, dim=1).view(-1, 1)
        if self.params.wt_scaling==0:
            weight = (weight+1.0)/2.0
        elif self.params.wt_scaling>0:
            weight = sigmoid(self.params.wt_scaling*weight)    
        #weights = weights.cpu().data
        self.weight = weight
        # logger.info("total loss:")
        # logger.info(to)
        return weight
    
    def lm_dis_weight(self, stats):
        loss_fn = WeightedMSELoss(size_average=False)
        x, y, _ = self.get_lm_xy(volatile=False)#volatile=False
        x = Variable(x)
        y = Variable(y, requires_grad=False) #,
        sigmoid = Sigmoid()
        mapped = self.mapping(x)
        c_x = mapped / mapped.norm(2, 1, keepdim=True).expand_as(mapped)
        c_y = y / y.norm(2,1, keepdim=True).expand_as(y)
        weight = torch.sum(c_x * c_y, dim=1).view(-1,1)
        if self.params.wt_scaling==0:
            #logger.info(weights)
            weight = (weight+1.0)/2.0
            #weight = torch.clamp(weight,0,1)
        elif self.params.wt_scaling!=0:
            weight = sigmoid(self.params.wt_scaling*weight)

        #logger.info(weights)
        loss = loss_fn(mapped, y, weight)
        stats['DIS_COSTS'].append(loss.data[0])
        if (loss!=loss).data.any():
            logger.error("NaN detected")
            exit()

        self.mapping.zero_grad()
        loss.backward()
        self.m_optimizer.step()
        self.orthogonalize()

    def lm_dis_const_weight(self, stats):
        loss_fn = WeightedMSELoss(size_average=False)
        x, y, weight = self.get_lm_xy(volatile=False)#volatile=False
        x = Variable(x)
        y = Variable(y, requires_grad=False) #,
        mapped = self.mapping(x)

        #logger.info(weights)
        loss = loss_fn(mapped, y, weight)
        stats['DIS_COSTS'].append(loss.data[0])
        if (loss!=loss).data.any():
            logger.error("NaN detected")
            exit()

        self.mapping.zero_grad()
        loss.backward()
        self.m_optimizer.step()
        self.orthogonalize()

    def lm_dis(self, stats):
        loss_fn = torch.nn.MSELoss(size_average=False)
        #loss_fn = torch.nn.CosineEmbeddingLoss(size_average=False)
        x, y = self.get_lm_xy(volatile=False) # get_lm_xy
        x = Variable(x)
        y = Variable(y, requires_grad=False)

        #for i in range(self.params.dis_steps):
        mapped = self.mapping(x)
        loss = loss_fn(mapped, y)#, self.tt
        #loss = loss_fn(mapped, y, self.tt)#
        stats['DIS_COSTS'].append(loss.data[0])
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        self.mapping.zero_grad()
        loss.backward()
        self.m_optimizer.step()
        self.orthogonalize()

    def lm_dis_origin(self, stats):
        loss_fn = torch.nn.MSELoss(size_average=False)
        #loss_fn = torch.nn.CosineEmbeddingLoss(size_average=False)
        x, y = self.get_train_xy(volatile=False)
        x = Variable(x)
        y = Variable(y, requires_grad=False)

        #for i in range(self.params.dis_steps):
        mapped = self.mapping(x)
        loss = loss_fn(mapped, y)#, self.tt
        #loss = loss_fn(mapped, y, self.tt)#
        stats['DIS_COSTS'].append(loss.data[0])
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        self.mapping.zero_grad()
        loss.backward()
        self.m_optimizer.step()
        self.orthogonalize()

    def procrustes(self):
        """
        Find the best orthogonal matrix mapping using the Orthogonal Procrustes problem
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        """
        logger.info("Training with Procrustes")
        logger.info("dico size:"+str(len(self.dico)))
        A = self.src_emb.weight.data[self.dico[:, 0]]
        B = self.tgt_emb.weight.data[self.dico[:, 1]]
        W = self.mapping.weight.data
        M = B.transpose(0, 1).mm(A).cpu().numpy()
        U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
        W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))

    def orthogonalize(self):
        """
        Orthogonalize the mapping.
        """
        #logger.info("using orthogonalization")
        if self.params.map_beta > 0:
            W = self.mapping.weight.data
            beta = self.params.map_beta
            W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))

    def update_lr(self, to_log, metric):
        """
        Update learning rate when using SGD.
        """
        if 'sgd' not in self.params.map_optimizer:
            return
        old_lr = self.map_optimizer.param_groups[0]['lr']
        new_lr = max(self.params.min_lr, old_lr * self.params.lr_decay)
        if new_lr < old_lr:
            logger.info("Decreasing learning rate: %.8f -> %.8f" % (old_lr, new_lr))
            self.map_optimizer.param_groups[0]['lr'] = new_lr
        
        if self.params.lr_shrink < 1 and to_log[metric] >= -1e7:
            if to_log[metric] < self.best_valid_metric:
                logger.info("Validation metric is smaller than the best: %.5f vs %.5f"
                            % (to_log[metric], self.best_valid_metric))
                # decrease the learning rate, only if this is the
                # second time the validation metric decreases
                if self.decrease_lr:
                    old_lr = self.map_optimizer.param_groups[0]['lr']
                    self.map_optimizer.param_groups[0]['lr'] *= self.params.lr_shrink
                    logger.info("Shrinking the learning rate: %.5f -> %.5f"
                                % (old_lr, self.map_optimizer.param_groups[0]['lr']))
                self.decrease_lr = True
        
    def update_lr_dis(self, to_log, metric):
        """
        Update learning rate when using SGD.
        """
        logger.info("in dis update")
        old_lr = self.m_optimizer.param_groups[0]['lr']
        new_lr = max(self.params.min_lr, old_lr * self.params.lr_decay)
        if new_lr < old_lr:
            logger.info("Decreasing learning rate: %.8f -> %.8f" % (old_lr, new_lr))
            self.m_optimizer.param_groups[0]['lr'] = new_lr
            self.decrease_lr = True
        '''
        if self.params.lr_shrink < 1 and to_log[metric] >= -1e7:
            if to_log[metric] < self.best_valid_metric:
                logger.info("Validation metric is smaller than the best: %.5f vs %.5f"
                            % (to_log[metric], self.best_valid_metric))
                # decrease the learning rate, only if this is the
                # second time the validation metric decreases
                if self.decrease_lr:
                    old_lr = self.m_optimizer.param_groups[0]['lr']
                    self.m_optimizer.param_groups[0]['lr'] *= self.params.lr_shrink
                    logger.info("Shrinking the learning rate: %.5f -> %.5f"
                                % (old_lr, self.m_optimizer.param_groups[0]['lr']))
                self.decrease_lr = True
        '''

    def save_best(self, to_log, metric):
        """
        Save the best model for the given validation metric.
        """
        # best mapping for the given validation criterion
        if to_log[metric] > self.best_valid_metric:
            # new best mapping
            self.best_valid_metric = to_log[metric]
            logger.info('* Best value for "%s": %.5f' % (metric, to_log[metric]))
            # save the mapping
            W = self.mapping.weight.data.cpu().numpy()
            path = os.path.join(self.params.exp_path, 'best_mapping.pth')
            logger.info('* Saving the mapping to %s ...' % path)
            torch.save(W, path)

    def reload_best(self):
        """
        Reload the best mapping.
        """
        path = os.path.join(self.params.exp_path, 'best_mapping.pth')
        logger.info('* Reloading the best model from %s ...' % path)
        # reload the model
        assert os.path.isfile(path)
        to_reload = torch.from_numpy(torch.load(path))
        W = self.mapping.weight.data
        assert to_reload.size() == W.size()
        W.copy_(to_reload.type_as(W))

    def export(self):
        """
        Export embeddings.
        """
        params = self.params

        # load all embeddings
        logger.info("Reloading all embeddings for mapping ...")
        params.src_dico, src_emb = load_embeddings(params, source=True, full_vocab=True)
        params.tgt_dico, tgt_emb = load_embeddings(params, source=False, full_vocab=True)

        # apply same normalization as during training
        normalize_embeddings(src_emb, params.normalize_embeddings, mean=params.src_mean)
        normalize_embeddings(tgt_emb, params.normalize_embeddings, mean=params.tgt_mean)

        # map source embeddings to the target space
        bs = 4096
        logger.info("Map source embeddings to the target space ...")
        for i, k in enumerate(range(0, len(src_emb), bs)):
            x = Variable(src_emb[k:k + bs], volatile=True)
            src_emb[k:k + bs] = self.mapping(x.cuda() if params.cuda else x).data.cpu()

        # write embeddings to the disk
        export_embeddings(src_emb, tgt_emb, params)

class WeightedMSELoss(MSELoss):
    def __init__(self, size_average=True, reduce=True):
        super(WeightedMSELoss, self).__init__(size_average)
        self.reduce = reduce

    def forward(self, input, target, weight):
        '''
        logger.info("input shape:")
        logger.info(type(input))
        logger.info("target shape:")
        logger.info(type(target))
        logger.info("weight:")
        logger.info(type(weight))
        '''
        return torch.sum(torch.autograd.Variable(weight)*(input-target)**2)
        #return weighted_mse_loss(input, target, weight, size_average=self.size_average, reduce=self.reduce)
'''
def weighted_mse_loss(input, target, weight, size_average=False, reduce=True):
    #loss =  (input-target)**2
    loss = (input-target)**2
    # logger.info(loss)
    #if not reduce:
    #    return loss
    if size_average:
        return torch.mean(loss)
    return torch.sum(loss)
'''
