#!/usr/bin/python3
# -*- coding: utf-8 -*-

import epub
import pickle
from pprint import pprint
from lxml import etree
import re
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
#print('\n'.join(txt[:-3]))
txt=txt[:-3]
with open('data/kaate_dikt.pickle',mode='wb') as ff:
    pickle.dump(txt,ff,protocol=2)

book=epub.open('data/Min_kamp.epub')

txt=[]

rr_css = re.compile('(?:\s*\S+\s*{[^}]*})+')
notset=set(['\n','\n\r','\r\n','',' ','\t',None])

re_signs=re.compile('([\.\-\+#@_\/\,\:\;\?\!"\'\(\)\}\{\[\]])',re.UNICODE|re.MULTILINE)
re_multispace=re.compile('[ ]+')


def get_text(cc,addList,level=0):
    for c in cc.iterchildren():
        if c.text is not None:
#            if rr_css.findall(c.text):
#                continue
            outtxt=re_signs.sub(r' \1 ',c.text)
            outtxt=re_multispace.sub(' ',outtxt)
            if len(outtxt)>3:
                addList.append(outtxt)
            #print(outtxt,c,cc)
            #input(level)
        get_text(c,addList,level=level+1)

for iii,item in enumerate(book.opf.manifest.values()):
    try:
        # read the content
        data = book.read_item(item)

        tree=etree.fromstring(data)
        get_text(tree,txt)
        #pprint(data)
        #pprint(iii)
        #input('')
            #pprint(tree)


                #print(child2.text)
        #input('new tree')
    except Exception as e:
        print(e)
        #print(data)
        #input('exception')

print(len(txt))
print('\n'.join(txt[-31:-25]))
#print('\n'.join(txt[21:25]))
txt=txt[21:-25]
dd={}
dd['raw_text_list']=txt



dd['sentencelist']=[]
re_sentence = re.compile('([\.\?\!\n])',re.UNICODE|re.MULTILINE)
re_sentence_split=re.compile('SENTENCESPLIT',re.UNICODE|re.MULTILINE)


for iii,t1 in enumerate(txt):
    print(t1)
    t2=re_sentence.sub(r'\1 SENTENCESPLIT',t1)
    sentences=re_sentence_split.split(t2)

    for s in sentences:
        ssplit=s.split()
        ssplit=[k.lower() for k in ssplit]
        dd['sentencelist'].append(ssplit)

print(len(dd['sentencelist']))


with open('data/min_kamp_-_andre_bok.pickle',mode='wb') as ff:
    pickle.dump(dd,ff,protocol=2)
