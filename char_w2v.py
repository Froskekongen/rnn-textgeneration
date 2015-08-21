#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pickle
import gensim, logging
with open('data/tweets_list.pickle',mode='rb') as ff:
    tweets=pickle.load(ff)

tweetchars=[list(t.lower()) for t in tweets]

print(len(tweets))
print(tweetchars[0])



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model = gensim.models.Word2Vec(tweetchars,size=8, window=25, min_count=5, workers=5)


print(model['a'])
print(model['e'])
print(model['?'])
print(model.most_similar(positive=['a']))
print(model.most_similar(positive=['e']))
print(model.most_similar(positive=['?']))

model.save('models/word2vec_model.model')
