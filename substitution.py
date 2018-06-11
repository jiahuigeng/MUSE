import gzip
import argparse
line_cnt =0
number_cnt = 0
url_cnt = 0
new_content = []


if __name__== "__main__":
    argparser = argparse.ArgumentParser(description='Evaluation')
    argparser.add_argument("--ref", required=True, type=str, help="categ file with concret number and url")
    argparser.add_argument("--trans", required=True, type=str, help="translation with  $number and $url")
    argparser.add_argument("--output",required=True, type=str, help="output file with full path")
    params = argparser.parse_args()
    with gzip.open(params.ref,'rt') as f1, open(params.trans,'rt') as f2:
        #with open("/work/smt2/jgeng/MUSE/dumped/4a5v78lrj6/sent-translation.en",'rt') as f1:
        for l1, l2 in zip(f1,f2):
            l1, l2 = l1.split(), l2.split()
            l1_dict, l2_dict = [],[]

            for i in range(len(l1)):
                if l1[i]=="$number" or l1[i]=="$url":
                    for j in range(len(l2)):
                        if l2[j]==l1[i]:
                            l2[j]=l1[i+2]
                            break

            new_content.append(' '.join(l2))


    with open(params.output, 'wt') as f3:
        for item in new_content:
            f3.write("{}\n".format(item))
    print("hello world")