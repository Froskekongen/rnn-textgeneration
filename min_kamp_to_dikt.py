from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file
from keras.utils import np_utils, generic_utils
import numpy as np
import random, sys
import codecs
import pickle
from random import shuffle
from copy import copy

import re

mult_newline=re.compile('[\n]{2,}',flags=re.UNICODE|re.MULTILINE)

'''
    Example script to generate text from Nietzsche's writings.

    At least 20 epochs are required before the generated text
    starts sounding coherent.

    It is recommended to run this script on GPU, as recurrent
    networks are quite computationally intensive.

    If you try this script on new data, make sure your corpus
    has at least ~100k characters. ~1M is better.
'''


# with open('/home/erlenda/.keras/datasets/tweets.pickle',mode='rb') as ff:
#     text=pickle.load(ff)


with open('./data/kaate_dikt.pickle',mode='rb') as ff:
    text2=pickle.load(ff)
    text2='\n'.join(text2)
    text2=text2.replace(' .','.')
    text2=text2.replace(' ,',',')
    text2=text2.replace(' :',':')
    text2=text2.replace(' ?','?')
    text2=text2.replace(' !','!')
    text2=text2.replace(' ;',';')
    text2=text2.replace('\r','')
    text2=mult_newline.sub('\n\n',text2)


with open('./data/min_kamp_-_andre_bok.pickle',mode='rb') as ff:
    text=pickle.load(ff)
    text='\n\n'.join(text['raw_text_list'])
    text=text.replace(' .','.')
    text=text.replace(' ,',',')
    text=text.replace(' :',':')
    text=text.replace(' ?','?')
    text=text.replace(' !','!')
    text=text.replace(' ;',';')
    text=text.replace('\r','')
    text=mult_newline.sub('\n\n',text)



print('corpus length:', len(text))
text=text.lower()
text2=text2.lower()

chars = set(text+text2)
print(list(sorted(chars)))
print('total chars:', len(chars))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentenceCont=[]
nc_cont=[]
for jjj in range(step):
    sentences=[]
    next_chars=[]
    for i in range(jjj, len(text) - maxlen, step):
        sentences.append(text[i : i + maxlen])
        next_chars.append(text[i + maxlen])
    sentenceCont.append(sentences)
    nc_cont.append(next_chars)




sentenceCont2=[]
nc_cont2=[]
for jjj in range(step):
    sentences=[]
    next_chars=[]
    for i in range(jjj, len(text2) - maxlen, step):
        sentences.append(text2[i : i + maxlen])
        next_chars.append(text2[i + maxlen])
    sentenceCont2.append(sentences)
    nc_cont2.append(next_chars)


print('nb sequences:', len(sentences))
print(sentences[0])
print(next_chars[0])
print('\n'*5)
print(sentences[1])
print(next_chars[1])
print('\n'*5)
print(sentences[50])
print(next_chars[50])




layerdims=768*2
layerdims2=int(layerdims/5)
# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
## LSTM layers
model.add(LSTM(len(chars), layerdims, return_sequences=True))
model.add(Dropout(0.15))
model.add(LSTM(layerdims, layerdims, return_sequences=False))
model.add(Dropout(0.15))

model.add(Dense(layerdims, layerdims2))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(layerdims2, len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# helper function to sample an index from a probability array
def sample(a, temperature=1.0):
    a = np.log(a)/temperature
    a = np.exp(a)/np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1,a,1))

import codecs
# train the model, output generated text after each iteration




for iteration in range(1, 140):
    it1=iteration
    print()
    print('-' * 50)
    print('Iteration', iteration)
    if iteration==50:
        sentenceCont=sentenceCont2
        nc_cont=nc_cont2
    sentences=sentenceCont[iteration%3]
    next_chars=nc_cont[iteration%3]

    progbar = generic_utils.Progbar(len(sentences))
    start=0
    batch_size=128
    while 1:
        loc_sentences=sentences[start:start+batch_size]
        loc_next_chars=next_chars[start:start+batch_size]
        start=start+batch_size
        max_ss=len(loc_sentences)
        if max_ss==0:
            break
        X = np.zeros((max_ss, maxlen, len(chars)), dtype=np.bool)
        y = np.zeros((max_ss, len(chars)), dtype=np.bool)
        for index in range(max_ss):
            sentence=loc_sentences[index]
            nc=loc_next_chars[index]
            for t, char in enumerate(sentence):
                X[index, t, char_indices[char]] = 1
            y[index, char_indices[nc]] = 1

        loss = model.train_on_batch(X,y)
        progbar.add(X.shape[0], values=[("train loss", loss)])
        del X,y
    model.save_weights('models/minkamp_dikt2_'+str(it1)+'.hdf5')
    list1_shuf = []
    list2_shuf = []
    index_shuf = list(range(len(sentences)))
    shuffle(index_shuf)
    for i in index_shuf:
        list1_shuf.append(sentences[i])
        list2_shuf.append(next_chars[i])
    sentences=copy(list1_shuf)
    next_chars=copy(list2_shuf)


    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 0.7, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index : start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for iteration in range(2000):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        with codecs.open('data/generated_mkdikt2_'+str(it1)+'_'+str(diversity)+'.txt',mode='w',encoding='utf-8') as ff:
            ff.write(generated)
        print()
