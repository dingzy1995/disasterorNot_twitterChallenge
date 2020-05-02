# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:48:15 2020

@author: zding
"""

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import preprocessing
import numpy as np

lda = LatentDirichletAllocation(random_state=0)
df=preprocessing.as_list('train.csv')
data=df['text_list'].tolist()
data_list=[]
for l in data:
    for token in l:
        data_list.append(token)
count_vect = CountVectorizer(stop_words='english')
doc_terms=count_vect.fit_transform(data_list)

lda.fit(doc_terms)

for i, topic in enumerate(lda.components_):
    print([count_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')
