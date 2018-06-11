import os
import argparse
from collections import OrderedDict

from src.utils import bool_flag, initialize_exp
from src.models import build_model
from src.trainer import Trainer
from src.evaluation.translator import Translator
from src.g_utils import load_translation_data, load_vocabulary

import kenlm


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
    parser.add_argument("--exp_path", type=str, default="exps", help="Where to store experiment logs and models")
    parser.add_argument("--exp_name", type=str, default="a", help="Where to store experiment logs and models")
    parser.add_argument("--exp_id", type=str, default="", help="Where to store experiment logs and models")

    parser.add_argument("--cuda", type=bool_flag, default=False, help="Run on GPU")
    # data
    parser.add_argument("--src_lang", type=str, default="de", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="en", help="Target language")
    # reload pre-trained embeddings
    parser.add_argument("--src_emb", required=True, type=str, help="Reload source embeddings")
    parser.add_argument("--tgt_emb", required=True, type=str, help="Reload target embeddings")
    parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size")
    parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
    parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")

    # added
    parser.add_argument("--src_file", required=True, type=str, help="Source corpus to translate")
    parser.add_argument("--tgt_file", required=True, type=str, help="Reference corpus to evaluate")
    parser.add_argument("--n_sents", type=int, default=6000000, help="Number of sentences to translate")

    parser.add_argument("--src_vocab", type=str, default=None, help="Source vocab file")
    parser.add_argument("--tgt_vocab", type=str, default=None, help="Target vocab file")
    parser.add_argument("--method", type=str, default='csls', help="Target vocab file")

    parser.add_argument("--lm",type=str, default="", help="Target language model file")
    parser.add_argument('--lm_scaling', type=float, default=0.1, help="Language model scaling")
    parser.add_argument('--lex_scaling', type=float, default=1.0, help="lexicon model scaling")
    parser.add_argument('--topk', type=int, default=100, help='top k candidates for word candidates')
    parser.add_argument('--keep', type=bool_flag, default=True, help='keep the original word for unknown case')
    parser.add_argument('--beam_size', type=int, default=10, help="beam size for candidates")
    # parse parameters
    params = parser.parse_args()

    # check parameters
    assert params.src_lang, "source language undefined"
    assert os.path.isfile(params.src_emb)
    assert not params.tgt_lang or os.path.isfile(params.tgt_emb)

    # build logger / model / trainer / evaluator
    logger = initialize_exp(params)
    src_emb, tgt_emb, mapping, _ = build_model(params, False)
    data = load_translation_data(params.src_lang, params.tgt_lang, params.src_file, params.tgt_file, params.n_sents)
    vocab = None
    lm = ""
    if params.lm!="":
        lm = kenlm.LanguageModel(params.lm)



    if params.tgt_vocab and params.tgt_vocab:
        vocab = load_vocabulary(params.src_lang,params.src_vocab, params.tgt_lang, params.tgt_vocab)

    trainer = Trainer(src_emb, tgt_emb, mapping, None, params)
    translator = Translator(trainer)


    # run evaluations
    to_log = OrderedDict({'n_iter': 0})
    if params.tgt_lang:
        ####keep
        #print(translator.word_translation(to_log,'der',params.src_lang, params.tgt_lang, 'csls', topk=10))
        #print(translator.word_translation(to_log,'das',params.src_lang, params.tgt_lang, 'csls', topk=10))
        #print(translator.word_translation(to_log,'die',params.src_lang, params.tgt_lang, 'csls', topk=10))
        translator.corpus_translation_w(data, params.src_lang, params.tgt_lang,to_log, 'csls', topk=params.topk, lm=lm, lm_scaling=params.lm_scaling, lex_scaling=params.lex_scaling,keep=params.keep, beam_size=params.beam_size)
        #print(translator.sent_translation_w(['was','is','in','of','as','the','to','in','and','on'],to_log,params.src_lang, params.tgt_lang, 'csls'))
        #print(translator.sent_translation_w(['der','die','das','diese'],to_log,params.src_lang, params.tgt_lang, 'csls'))
        #print(translator.sent_translation_w(['jahr', 'tag', 'welt', 'zeit', 'leben', 'menschen', 'mann', 'moment', 'weise', 'frau'],to_log,'de','en', 'csls'))
