# -*- coding: utf-8 -*-
"""
@author: zding
"""

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import preprocessing
import numpy as np
import pyLDAvis
import pyLDAvis.sklearn

lda = LatentDirichletAllocation(n_components=10, max_iter=5)
df=preprocessing.as_list('train.csv')
data=df['text_list'].tolist()
disaster_data=[]
nondisaster_data=[]
for index, row in df.iterrows():
    new_list=[]
    for t in row['text_list']:
        if not t.startswith('http') and \
            not t.startswith('#') and \
            not t.startswith('@'):
                new_list.append(t)
    if row['target']==1:
        disaster_data.append(','.join(new_list))
    else:
        nondisaster_data.append(','.join(new_list))
count_vect = CountVectorizer(stop_words='english')
'''
edit here if want to switch between nondisaster topics and disaster topics
'''
disaster_terms=count_vect.fit_transform(disaster_data)
#nondisaster_terms=count_vect.fit_transform(nondisaster_data)


lda.fit_transform(disaster_terms)

#print words in top ten topics
for index, topic in enumerate(lda.components_):
    print([count_vect.get_feature_names()[i] for i in topic.argsort()[-11:-1]])
    print('\n')
    
#visualize topic modeling using pyldavis
img=pyLDAvis.sklearn.prepare(lda, disaster_terms, count_vect)
pyLDAvis.show(img)
