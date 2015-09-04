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

'''
    Example script to generate text from Nietzsche's writings.

    At least 20 epochs are required before the generated text
    starts sounding coherent.

    It is recommended to run this script on GPU, as recurrent
    networks are quite computationally intensive.

    If you try this script on new data, make sure your corpus
    has at least ~100k characters. ~1M is better.
'''


with open('/home/erlenda/.keras/datasets/tweets.pickle',mode='rb') as ff:
    text=pickle.load(ff)
print('corpus length:', len(text))
text=text[:2000000].lower()

chars = set(text)
print(list(sorted(chars)))
print('total chars:', len(chars))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 25
step = 3
N_minib=int(  (len(text)-maxlen)/step  )
minib_size=128




print('Vectorization...')
X = np.zeros((minib_size, maxlen, len(chars)), dtype=np.bool)
y = np.zeros((minib_size, len(chars)), dtype=np.bool)
def get_minibdata(tt,maxlength,space,X,y,minibatch_number=0,minibatch_size=128):
    sentences=[]
    nc=[]
    start=space*minibatch_number*minibatch_size
    ending=0
    N_minibatches=int( (len(tt)-maxlength)/space )
    if minibatch_number>=N_minibatches:
        return None,None
    for iii in range(start,start+space*minibatch_size,space):
        sentences.append(tt[iii:iii+maxlength])
        nc.append(tt[iii+maxlength])
        ending=iii+maxlength
    # for iii in range(0,5*5,5):
    #     print(sentences[iii])
    #     print('Next char: ',nc[iii])
    #     print('\n'*5)
    # print(len(sentences),start,ending)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[nc[i]]] = 1

get_minibdata(text,maxlen,step,X,y,minibatch_number=0,minibatch_size=minib_size)
print(y[0])

#sys.exit(0)

layerdims=64
# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(len(chars), layerdims, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(layerdims, layerdims, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(layerdims, len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')

# helper function to sample an index from a probability array
def sample(a, temperature=1.0):
    a = np.log(a)/temperature
    a = np.exp(a)/np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1,a,1))

import codecs
# train the model, output generated text after each iteration

nb_epoch=60
for e in range(nb_epoch):
    print('-'*40)
    print('Epoch', e)
    print('-'*40)
    print("Training...")
    # batch train with realtime data augmentation
    progbar = generic_utils.Progbar(X.shape[0]*512)
    for iii in range(512):
        get_minibdata(text,maxlen,step,X,y,minibatch_number=iii,minibatch_size=minib_size)

        loss = model.train_on_batch(X, y)
        progbar.add(X.shape[0], values=[("train loss", loss)])

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index : start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for iteration in range(1000):
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
        with codecs.open('generated_'+str(e)+'_'+str(diversity)+'.txt',mode='w',encoding='utf-8') as ff:
            ff.write(generated)
        print()
