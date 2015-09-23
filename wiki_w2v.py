#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pickle
import gensim, logging
if __name__ == '__main__':

    with open('./data/wikipages/sentlist.pickle',mode='rb') as ff:
        wiki_sents=pickle.load(ff)

    print(wiki_sents[0].lower().split())
    token_sents=[ws.lower().split() for ws in wiki_sents]
    print(token_sents[0])


    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model = gensim.models.Word2Vec(token_sents,size=200, window=5, min_count=200, workers=6,negative=6)


    print(model.most_similar(positive=['dessuten']))
    print(model.most_similar(positive=['.']))
    print(model.most_similar(positive=['?']))
    print(model.most_similar(positive=['skog']))
    print(model.most_similar(positive=['og']))
    print(model.most_similar(positive=['kriminell']))

    model.save('models/word2vec_wiki.model')
