from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file
import numpy as np
import random, sys
import codecs
import pickle

import theano

import gensim
import pdb

'''
Using word2vec first to represent characters. This makes character representation
continuous.
Then use this representation in LSTM
'''



with open('/home/erlenda/.keras/datasets/tweets.pickle',mode='rb') as ff:
    text=pickle.load(ff)
print('corpus length:', len(text))
text=text[:500000].lower()

#loading word2vec model
gs_model=gensim.models.Word2Vec.load('models/word2vec_model.model')

chars = set(text)
print(list(sorted(chars)))
print('total chars:', len(chars))

repDim=gs_model['e'].shape[0]

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 25
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i : i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))
print(sentences[0])
print(next_chars[0])
print('\n'*5)
print(sentences[1])
print(next_chars[1])
print('\n'*5)
print(sentences[50])
print(next_chars[50])



print('Vectorization...:',len(sentences),maxlen)
X = np.zeros((len(sentences), maxlen, repDim), dtype=theano.config.floatX)
y = np.zeros((len(sentences), repDim), dtype=theano.config.floatX)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i,t,:]=gs_model[char]
        #X[i, t, char_indices[char] = 1
    #y[i, char_indices[next_chars[i]]] = 1
    y[i, :] = gs_model[next_chars[i]]
    sentences[i]=None
print(y[:5,:])

layerdims=512
# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(repDim, layerdims, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(layerdims, layerdims, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(layerdims, layerdims, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(layerdims, repDim))
model.add(Activation('linear'))

model.compile(loss='mean_squared_error', optimizer='rmsprop')

# helper function to sample an index from a probability array
# def sample(a, temperature=1.0):
#     a = np.log(a)/temperature
#     a = np.exp(a)/np.sum(np.exp(a))
#     return np.argmax(np.random.multinomial(1,a,1))

## replacing sampling function - word2vec gives closeness to a specific character
# idea: sample based on nearest character.

import codecs
# train the model, output generated text after each iteration
for iteration in range(1, 60):
    it1=iteration
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=1)

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.1, 0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index : start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated+'\n'*5)

        for iteration in range(1000):
            x = np.zeros((1, maxlen, repDim))
            for t, char in enumerate(sentence):
                x[0, t, :] = gs_model[char]


            preds = model.predict(x, verbose=0)[0]
            #print(type(preds),preds)
            next_chars=gs_model.most_similar(positive=[preds], topn=100)

            distr=np.array([nc[1] for nc in next_chars])
            distr=np.array([d*np.exp(diversity*iii) for iii,d in enumerate(distr)])
            distr=distr/distr.sum()
            mmax=np.argmax(np.random.multinomial(1,distr,1))
            next_char=next_chars[mmax][0]
            #print(next_char)
            #next_index = sample(preds, diversity)
            #next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        with codecs.open('generated_'+str(it1)+'_'+str(diversity)+'.txt',mode='w',encoding='utf-8') as ff:
            ff.write(generated)
        print()
