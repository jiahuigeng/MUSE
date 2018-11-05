# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import time
import json
import argparse
from collections import OrderedDict
import numpy as np
import torch
import gzip
import kenlm
from src.utils import bool_flag, initialize_exp
from src.models import build_model
from src.g_trainer import Trainer
from src.evaluation import Evaluator
from src.g_utils import load_translation_data
from src.evaluation.translator import Translator
from src.utils import load_mapped_embeddings, normalize_embeddings
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
VALIDATION_METRIC = 'mean_cosine-csls_knn_10-S2T-10000'


# main
parser = argparse.ArgumentParser(description='Unsupervised training')
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
parser.add_argument("--export", type=str, default="txt", help="Export embeddings after training (txt / pth)")
# data
parser.add_argument("--src_lang", type=str, default='de', help="Source language")
parser.add_argument("--tgt_lang", type=str, default='en', help="Target language")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--max_vocab", type=int, default=50000, help="Maximum vocabulary size (-1 to disable)")
# mapping
parser.add_argument("--map_id_init", type=bool_flag, default=True, help="Initialize the mapping as an identity matrix")
parser.add_argument("--map_beta", type=float, default=0.001, help="Beta for orthogonalization")
# discriminator
parser.add_argument("--dis_layers", type=int, default=2, help="Discriminator layers")
parser.add_argument("--dis_hid_dim", type=int, default=2048, help="Discriminator hidden layer dimensions")
parser.add_argument("--dis_dropout", type=float, default=0., help="Discriminator dropout")
parser.add_argument("--dis_input_dropout", type=float, default=0.1, help="Discriminator input dropout")
parser.add_argument("--dis_steps", type=int, default=3, help="Discriminator steps")
parser.add_argument("--dis_lambda", type=float, default=1, help="Discriminator loss feedback coefficient")
parser.add_argument("--dis_most_frequent", type=int, default=75000, help="Select embeddings of the k most frequent words for discrimination (0 to disable)")
parser.add_argument("--dis_smooth", type=float, default=0.1, help="Discriminator smooth predictions")
parser.add_argument("--dis_clip_weights", type=float, default=0, help="Clip discriminator weights (0 to disable)")
# training adversarial
parser.add_argument("--adversarial", type=bool_flag, default=True, help="Use adversarial training")
parser.add_argument("--n_epochs", type=int, default=5, help="Number of epochs")
parser.add_argument("--epoch_size", type=int, default=100000, help="Iterations per epoch")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--map_optimizer", type=str, default="sgd,lr=0.1", help="Mapping optimizer")
parser.add_argument("--dis_optimizer", type=str, default="sgd,lr=0.1", help="Discriminator optimizer")
parser.add_argument("--lr_decay", type=float, default=0.5, help="Learning rate decay (SGD only)")
parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (SGD only)")
parser.add_argument("--lr_shrink", type=float, default=0.5, help="Shrink the learning rate if the validation metric decreases (1 to disable)")
# training refinement

parser.add_argument("--src_file", type=str, default="data/newstest2016.pp.fc.de-en.de.lc", help="Source corpus to translate")
parser.add_argument("--tgt_file", type=str, default="lm-jgeng/trans/0/sent-translation.en", help="Target file")

parser.add_argument("--n_refinement", type=int, default=0, help="Number of refinement iterations (0 to disable the refinement procedure)")
# dictionary creation parameters (for refinement)
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
parser.add_argument("--dico_build", type=str, default='S2T&T2S', help="S2T,T2S,S2T|T2S,S2T&T2S")
parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
parser.add_argument("--dico_max_rank", type=int, default=15000, help="Maximum dictionary words rank (0 to disable)")
parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
# reload pre-trained embeddings
parser.add_argument("--src_emb", type=str, default="", help="Reload source embeddings")
parser.add_argument("--tgt_emb", type=str, default="", help="Reload target embeddings")

parser.add_argument("--mapped_src_emb", type=str, default="", help="mapped source embeddings")
parser.add_argument("--mapped_tgt_emb", type=str, default="", help="mapped target embeddings")

parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")

# added by gjh
parser.add_argument("--src_range", type=str, default="", help="embedding start index")
parser.add_argument("--tgt_range", type=str, default="", help="embedding start index")
parser.add_argument("--n_lm_dis", type=int, default=0, help="Number of lm discriminative training (default 5, 0 to disable)")
parser.add_argument("--n_nep", type=int, default=10, help="Number of NN epoches on one dictionary")
parser.add_argument("--rev", type=bool_flag, default=False, help="Reverse Mode")
parser.add_argument("--sent_batch", type=int, default=5000, help="Number of sentences for each iteration")

parser.add_argument("--lm_train", type=bool_flag, default=False, help="Using LM training")
parser.add_argument("--procu", type=bool_flag, default=False, help="Using Procrustes training")
parser.add_argument("--lm",type=str, default="../corpus/pap.lm.en.5gram.trie", help="Target language model file")
parser.add_argument('--lm_scaling', type=float, default=0.1, help="Language model scaling")
parser.add_argument('--lex_scaling', type=float, default=1.0, help="lexicon model scaling")
parser.add_argument('--topk', type=int, default=100, help='top k candidates for word candidates')
parser.add_argument('--keep', type=bool_flag, default=True, help='keep the original word for unknown case')
parser.add_argument('--beam_size', type=int, default=10, help="beam size for candidates")
parser.add_argument("--reload", type=bool_flag, default=False, help="Load the pretrained network")
parser.add_argument("--n_map_dis", type=int, default=0, help="dictionary from embedding induction")
parser.add_argument("--init_lr", type=float, default=1e-2, help="initial learning rate")
parser.add_argument("--lr_thd", type=float, default=0.9, help="ratio for decrease the learning rate")
parser.add_argument("--dico_train", type=str, default="default", help="Path to training dictionary (default: use identical character strings)")
parser.add_argument("--total_sent", type=int, default="50000000", help="total sentences for training")
parser.add_argument("--last_lr", type=bool_flag, default=False, help="start from last learning rate")
parser.add_argument("--eval", type=bool_flag, default=False, help="evaluate the cross-lingual model")
parser.add_argument("--both_dico", type=bool_flag, default=False, help="use both dictionary")
parser.add_argument("--wt_scaling", type=int, default=-1, help="0 for linear scaling, 1 for sigmoid")
parser.add_argument("--const_lr", type=bool_flag, default=False, help="set true to use the const learning rate")


parser.add_argument("--ransac", type=bool_flag, default=False, help="whether to use ransac algorithm")
parser.add_argument("--ransac_n", type=int, default=600, help="number of pairs to train the mapping")
parser.add_argument("--ransac_k", type=int, default=100, help="maximum iterations for ransac procudure")
parser.add_argument("--ransac_t", type=int, default=500, help="threshold value for inliers")
parser.add_argument("--ransac_d", type=float, default=0.3, help="ratio for acceptable criterion")

parser.add_argument("--freq_procu", type=bool_flag, default=False, help="use frequcy based filtering on dico")
parser.add_argument("--freq_dis", type=bool_flag, default=10, help="")
parser.add_argument("--freq_th", type=int, default=10, help="minimum frequency for dico")
parser.add_argument("--freq_op", type=int, default=-1, help="dictionary operation, 0: src, 1:tgt, 2:and, 3:or ")
parser.add_argument("--src_corpus", type=str, default="", help="reference source corpus")
parser.add_argument("--tgt_corpus", type=str, default="", help="reference target corpus")

#parser.add_argument("--lm", type=str, required="n_lm_pir" or "n_lm_dis", help="the language model path")
# parser.add_argument("--lm", type=str, default="")
# parse parameters
params = parser.parse_args()

# check parameters
assert not params.cuda or torch.cuda.is_available()
assert 0 <= params.dis_dropout < 1
assert 0 <= params.dis_input_dropout < 1
assert 0 <= params.dis_smooth < 0.5
assert params.dis_lambda > 0 and params.dis_steps > 0
assert 0 < params.lr_shrink <= 1
assert os.path.isfile(params.src_emb)
assert os.path.isfile(params.tgt_emb)
assert params.dico_eval == 'default' or os.path.isfile(params.dico_eval)
assert params.export in ["", "txt", "pth"]

# build model / trainer / evaluator
logger = initialize_exp(params)
src_emb, tgt_emb, mapping, discriminator = build_model(params, True)
trainer = Trainer(src_emb, tgt_emb, mapping, discriminator, params)
evaluator = Evaluator(trainer)
lm = ""
if params.lm!="":
    logger.info("Loading the LM")
    lm = kenlm.LanguageModel(params.lm)
logger.info("LM loaded")
if params.reload==True:
    trainer.reload_best()
# logger.info(mapping.weight.data)
to_log = OrderedDict({'n_iter': 1})
evaluator.all_eval(to_log)
translator = Translator(trainer)


src_data = load_translation_data(params.src_file)
tgt_data = load_translation_data(params.tgt_file)


def get_iterator(iteratable, batch_size=1000, shuffle=False, repeat=False):
    if shuffle==False:
        cur_batch = []
        for item in iteratable:
            cur_batch.append(item.rstrip().split())
            if len(cur_batch) == batch_size:
                #print (cur_batch)
                yield cur_batch
                cur_batch = []
        if cur_batch:
            yield  cur_batch

if params.n_lm_dis>0:
    logger.info('----> LM TRAINING <----\n\n')

    # training loop

    for n_epoch in range(params.n_lm_dis):
        logger.info('Starting LM training epoch %i...' % n_epoch)
        tic = time.time()
        n_words_proc = 0
        stats = {'DIS_COSTS': []}

        f = open(params.src_file)
        src_iter = get_iterator(f, batch_size=params.sent_batch)
        for i in range(int(params.total_sent/params.sent_batch)):
            try:
                src_data = next(src_iter)
                logger.info(src_data[0])
                _src_dico, _mapped_src_emb = load_mapped_embeddings(params, source=True)
                _tgt_dico, _mapped_tgt_emb = load_mapped_embeddings(params, source=False)
                translator.emb1 = _mapped_src_emb
                translator.emb2 = _mapped_tgt_emb
                translator.src_dico = _src_dico
                translator.tgt_dico = _tgt_dico
                translator.reload()


                #trainer.load_training_dico(params.dico_train)
                tgt_data = translator.corpus_translation(src_data, method='csls', lm=lm)
                #logger.info(tgt_data[0])
                trainer.load_lm_dictionay(src_data, tgt_data)
                logger.info("dico size before filtering")
                logger.info(len(trainer.dico))
                if params.procu==True:
                    trainer.procrustes()
                    to_log = OrderedDict({'n_epoch': n_epoch})
                    evaluator.all_eval(to_log)
                    trainer.save_best(to_log, VALIDATION_METRIC)

                    loss_cnt = []
                    flag = True
                    if not params.last_lr:
                        trainer.m_optimizer.param_groups[0]['lr'] = params.init_lr
                    for sub_ep in range(params.n_nep):
                        # trainer.weight_distribution()

                        for n_iter in range(0, params.epoch_size, params.batch_size):
                            # discriminator training
                            for _ in range(params.dis_steps):
                                if params.wt_scaling==-1:
                                    trainer.lm_dis(stats)
                                else:
                                    trainer.lm_dis_const_weight(stats)

                            # log stats
                            if n_iter % 2000 == 0:
                                stats_str = [('DIS_COSTS', 'Discriminator loss')]
                                stats_log = ['%s: %.4f' % (v, np.mean(stats[k]))
                                             for k, v in stats_str if len(stats[k]) > 0]

                                stats_log.append('%i samples/s' % int(n_words_proc / (time.time() - tic)))
                                logger.info(('%06i - ' % n_iter) + ' - '.join(stats_log))
                                loss_cnt.append(float(stats_log[0].split()[-1]))

                                # reset
                                tic = time.time()
                                n_words_proc = 0
                                for k, _ in stats_str:
                                    del stats[k][:]

                        evaluator.all_eval(to_log)
                        evaluator.eval_dis(to_log)
                        logger.info(sum(loss_cnt)/len(loss_cnt))

                        trainer.update_lr_dis(to_log, VALIDATION_METRIC)

                        if params.const_lr:
                            pass
                        elif params.last_lr:
                            if trainer.m_optimizer.param_groups[0]['lr'] < params.min_lr:
                                trainer.m_optimizer.param_groups[0]['lr'] = params.min_lr

                        elif trainer.m_optimizer.param_groups[0]['lr'] < params.min_lr:
                            logger.info('Learning rate < 1e-6. BREAK.')
                            break

                        trainer.save_best(to_log, VALIDATION_METRIC)


                elif params.lm_train==True:
                    loss_cnt = []
                    flag = True
                    if not params.last_lr:
                        trainer.m_optimizer.param_groups[0]['lr'] = params.init_lr
                    for sub_ep in range(params.n_nep):
                        trainer.weight_distribution()
                        # if sub_ep==1:
                        #    for w in trainer.weight_distribution().numpy():
                        #        for i in w:
                        #            print(i)
                        for n_iter in range(0, params.epoch_size, params.batch_size):
                            # discriminator training
                            for _ in range(params.dis_steps):
                                if params.wt_scaling==-1:
                                    trainer.lm_dis(stats)
                                else:
                                    trainer.lm_dis_const_weight(stats)

                            # log stats
                            if n_iter % 2000 == 0:
                                stats_str = [('DIS_COSTS', 'Discriminator loss')]
                                stats_log = ['%s: %.4f' % (v, np.mean(stats[k]))
                                             for k, v in stats_str if len(stats[k]) > 0]

                                stats_log.append('%i samples/s' % int(n_words_proc / (time.time() - tic)))
                                logger.info(('%06i - ' % n_iter) + ' - '.join(stats_log))
                                loss_cnt.append(float(stats_log[0].split()[-1]))

                                # reset
                                tic = time.time()
                                n_words_proc = 0
                                for k, _ in stats_str:
                                    del stats[k][:]

                        evaluator.all_eval(to_log)
                        evaluator.eval_dis(to_log)
                        logger.info(sum(loss_cnt)/len(loss_cnt))

                        trainer.update_lr_dis(to_log, VALIDATION_METRIC)

                        if params.const_lr:
                            pass
                        elif params.last_lr:
                            if trainer.m_optimizer.param_groups[0]['lr'] < params.min_lr:
                                trainer.m_optimizer.param_groups[0]['lr'] = params.min_lr

                        elif trainer.m_optimizer.param_groups[0]['lr'] < params.min_lr:
                            logger.info('Learning rate < 1e-6. BREAK.')
                            break

                        trainer.save_best(to_log, VALIDATION_METRIC)

                trainer.reload_best()
                trainer.export()

            except StopIteration:
                break

        # embeddings / discriminator evaluation
        to_log = OrderedDict({'n_epoch': n_epoch})
        evaluator.all_eval(to_log)
        evaluator.eval_dis(to_log)

        # JSON log / save best model / end of epoch
        logger.info("__log__:%s" % json.dumps(to_log))

        logger.info('End of epoch %i.\n\n' % n_epoch)





#export embeddings
if params.eval:
    trainer.reload_best()
    evaluator = Evaluator(trainer)
    to_log = OrderedDict({'n_iter': 1})
    evaluator.all_eval(to_log)
    translator = Translator(trainer)
# if params.export:
#     trainer.reload_best()
#     trainer.export()
# # # #
