import argparse
import sys
from gensim.models.word2vec import Text8Corpus
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import time
import pickle
#
if __name__=="__main__":
    #####################################
    # functions:
    # 1 self phrase extraction
    # 2 save and load the phraser
    # 3 use the phraser to parse new dataset
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-input", type=str, help= "input file", required=True)
    argparser.add_argument("-saved_model", type=str, default="saved_model", help="filepath to save phraser model")
    argparser.add_argument("-test", action='store_false',help="flag if load phraser to the test data")
    argparser.add_argument("-phrase_model", required="-test" in sys.argv,type=str, help="phraser model")
    argparser.add_argument("-phrase_file", required="-test" in sys.argv,type=str, help="filepath to save the phrase")
    argparser.add_argument("-minCount",  default=5, type=int, help="minCount for word2phrase")
    argparser.add_argument("-threshold", default=100, type=int, help="threshold")
    params = argparser.parse_args()

    sentences = Text8Corpus(params.input)
    if  params.test:
        phrases =Phrases(sentences, min_count=params.minCount, threshold=params.threshold)
        pickle.dump(phrases, open(params.saved_model, "wb"))
        print("phrase model done at: %s " % params.saved_model)

    elif not params.test:
        phrases = pickle.load(open(params.phrase_model,"rb"))
        phraser = Phraser(phrases)
        # sent = [u'the', u'mayor', u'of', u'new', u'york', u'was', u'there']
        # # print(type(phraser))
        # print(phraser[sent])

        #sent = [u'the', u'mayor', u'of', u'new', u'york', u'was', u'there']
        #phraser(sent)
        with open(params.input) as sentences,  open(params.phrase_file, 'w') as f_out:
            for sent in sentences:
                print(type(sent))
                sent1 = phraser[sent.split()]
                f_out.write(' '.join(sent1)+'\n')
        print("phrase file done at: %s" % params.phrase_file)


    # train_corpus = "/work/smt2/jgeng/master_thesis/tr-en/en.wmt.comb.4-50.lc"
    #
    # start_t = time.time()
    # # sentences = Text8Corpus(train_corpus)
    # with open(train_corpus, 'r') as sentences:
    #     bigram = Phrases(sentences, threshold=100)
    #     pickle.dump(bigram, open("phrase_model", 'wb'))
    #     new_bigram = pickle.load(open("phrase_model", 'rb'))
    #
    #     # bigram = Phraser(bigram)
    #     test_sent = "that best-case scenario already means that wilshere is \
    #     certain to miss at least the next four england games and , given his history , \
    #     his involvement in next summer 's european championship is clearly uncertain ."
    #     test_sent = test_sent.split()
    #     print(new_bigram[test_sent])
    #     end_t = time.time()
    #     print("total time: ", end_t - start_t)
#

# import word2vec
# train_corpus = "/work/smt2/jgeng/master_thesis/tr-en/en.wmt.comb.4-50.lc"
# word2vec.word2phrase(train_corpus, '/work/smt2/jgeng/master_thesis/tr-en/en.wmt.comb.4-50-phrases', verbose=True)
#
