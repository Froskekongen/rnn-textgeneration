#!/usr/bin/python3
# -*- coding: utf-8 -*-

import epub
import pickle
from pprint import pprint
from lxml import etree
#tree = etree.parse('examples/feed.xml',encoding='utf8')


book=epub.open('data/K_te_dikt.epub')
txt=[]

try:
    for iii,item in enumerate(book.opf.manifest.values()):
        # read the content
        data = book.read_item(item)

        tree=etree.fromstring(data)
            #pprint(data)
            #pprint(tree)
        for t in tree.iterchildren():
            dikt=''
            for u in t.iterchildren():
                if u.text is not None:
                    if u.text!='Kåte Dikt':
                        dikt+=u.text+'\n'
            txt.append(dikt)
except Exception as e:
    print(e)
print('\n'.join(txt[:-3]))
txt=txt[:-3]
with open('data/kaate_dikt.pickle',mode='wb') as ff:
    pickle.dump(txt,ff,protocol=2)
