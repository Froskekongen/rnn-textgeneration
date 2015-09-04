#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pickle
import re



re_signs=re.compile('([\.\-\+#@_\/\,\:\;\?\!"\'\(\)\}\{\[\]])',re.UNICODE|re.MULTILINE)
re_multispace=re.compile('[ ]+')

re_sentence = re.compile('(?<=[\.\?\!\n])')
