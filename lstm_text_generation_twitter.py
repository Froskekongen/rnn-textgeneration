from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file
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
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
    sentences[i]=None
def get_minibdata(tt,maxlength,space,minibatch_number=0,minibatch_size=128):
    sentences=[]
    nc=[]
    start=space*N_minib*minibatch_number
    N_minibatches=
    for iii in range(start,start+space*N_minib,space):




layerdims=128
# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(len(chars), layerdims, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(layerdims, layerdims, return_sequences=True))
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


for epoch in range(1,60)

for iteration in range(1, 60):
    it1=iteration
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=1)

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
        with codecs.open('generated_'+str(it1)+'_'+str(diversity)+'.txt',mode='w',encoding='utf-8') as ff:
            ff.write(generated)
        print()
