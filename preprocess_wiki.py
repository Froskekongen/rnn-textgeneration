#!/usr/bin/python3
# -*- coding: utf-8 -*-

import codecs
import pickle
from collections import defaultdict
import sys
import bz2
import os.path
from pprint import pprint
import nltk

import gensim.corpora

import re

sent_detector = nltk.data.load('/home/erlenda/nltk_data/tokenizers/punkt/norwegian.pickle')

nl_re=re.compile('([\n]){2,}',flags=re.MULTILINE|re.UNICODE)
fnutt_replace=re.compile('([\']){2,}',flags=re.MULTILINE|re.UNICODE)

re_space_insert=re.compile('([\?\.\,\:\;\!\'="_$%&@#\(\)\[\]\{\}\*|«»])',flags=re.MULTILINE|re.UNICODE)
re_multispace=re.compile('[ ]+',flags=re.MULTILINE|re.UNICODE)

def preproc_for_tok(txt):
    txt=re_space_insert.sub(r' \1 ',txt)
    txt=re_multispace.sub(' ',txt)
    return txt

if __name__ == '__main__':
    """
    Usage:
    python3 preprocess_wiki.py /path/to/bz2/wikimedia/dump

    Output:
    xml pages of wiki (in data/wikipages)
    """
    print(sys.argv[1])
    if not os.path.isfile('./data/wikipages/wiki_pages.pickle'):
        print('joy1')
        texts = list([(text, title, pageid) for title, text, pageid in gensim.corpora.wikicorpus.extract_pages(bz2.BZ2File(sys.argv[1]))])
        with open('./data/wikipages/wiki_pages.pickle',mode='wb') as ff:
            pickle.dump(texts,ff,protocol=4)

    elif not os.path.isfile('./data/wikipages/wiki_pages_raw.pickle'):
        print('joy2')
        with open('./data/wikipages/wiki_pages.pickle',mode='rb') as ff:
            texts=pickle.load(ff)
        texts2=[]
        for text, title, pageid in texts:
            text=gensim.corpora.wikicorpus.filter_wiki(text)
            texts2.append((text,title,pageid))
        with open('./data/wikipages/wiki_pages_raw.pickle',mode='wb') as ff:
            pickle.dump(texts2,ff,protocol=4)

    elif not os.path.isfile('./data/wikipages/wiki_pages_raw_processed.pickle'):
        print('joy3')
        with open('./data/wikipages/wiki_pages_raw.pickle',mode='rb') as ff:
            texts=pickle.load(ff)

    # tt=nl_re.sub(r'\1\1',texts[4][0])
    # print(tt)
        texts2=[]
        for text, title, pageid in texts:
            if text.startswith('#REDIRECT'):
                continue
            text=nl_re.sub(r'\1\1',text)
            text=fnutt_replace.sub(r'\1',text)
            texts2.append((text,title,pageid))
        with open('./data/wikipages/wiki_pages_raw_processed.pickle',mode='wb') as ff:
            pickle.dump(texts2,ff,protocol=4)

    else:
        print('joy4')
        with open('./data/wikipages/wiki_pages_raw_processed.pickle',mode='rb') as ff:
            texts=pickle.load(ff)

    #print(text_preproc_for_tok(texts[1][0]))



    toklist=[]
    acc_sents=[]
    iii=0
    no_starts=['==','*','(','Kategori:','ISBN']

    def txt_startswith(txt,nostarts):
        low_st=txt[0].lower()
        for tt in nostarts:
            a=txt.startswith(tt)
            if a:
                return a
        if txt.startswith(low_st):
            return True
        if (txt.find('Kategori:') !=-1) or (txt.find('==') !=-1) or (txt.find('\n') !=-1):
            return True
        return False


    for text, title, pageid in texts:
        sents=sent_detector.tokenize(text)
        for sent in sents:
            if not ( txt_startswith(sent,no_starts) ):
                acc_sent=preproc_for_tok(sent)

                acc_sents.append(acc_sent)
            iii+=1
            if iii%5000==0:
                print(acc_sent)
                print('')
        # text=text_preproc_for_tok(text).lower()
        # toks=text.split()
        # toklist.extend(toks)

    with open('./data/wikipages/sentlist.pickle',mode='wb') as ff:
        pickle.dump(acc_sents,ff)
