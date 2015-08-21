#!/usr/bin/python3
# -*- coding: utf-8 -*-

import codecs
import pickle
from collections import defaultdict


sentences=[]
with codecs.open('data/ndt_1-0_nob.conll',encoding='utf-8') as ff:
    dd=defaultdict(list)
    for line in ff:
        if line=='\n':
            sentences.append(dd)
            dd=defaultdict(list)
            continue
        data=line.split('\t')
        dd['sentence'].append(data[1])
        dd['lemmas'].append(data[2])
        dd['pos1'].append(data[3])
        dd['pos2'].append(data[4])
        dd['intonasjon'].append(data[5])
        dd['depend'].append(int(data[6]))
        dd['funksjon'].append(data[7])

print(len(sentences))
print(sentences[0])

with open('data/ndt_nob.pickle',mode='wb') as ff:
    pickle.dump(sentences,ff)
