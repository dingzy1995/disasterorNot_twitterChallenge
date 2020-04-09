# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 17:43:19 2020

@author: zding
"""
"""
this script aims to perform a topic modeling and analysis on the disaster tweets
part of the code in this script is learnt from Shashank Kapadia's article
'Topic Modeling in Python: Latent Dirichlet Allocation (LDA)'
https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0

"""

import pandas as pd
import re
import argparse
from wordcloud import WordCloud

args = argparse.ArgumentParser(description='Program description.')
args.add_argument('-a','--address', default='train.csv', help='the address of the file to be process')
args = args.parse_args()

def read_data(address):
    df=pd.read_csv(address)
    #only need the tweet content for this task
    df.drop(columns=['id', 'keyword', 'location', 'target'], axis=1)
    #remove symbols
    df['text']=df['text'].map(lambda x: re.sub('\Z','',x))
    df['text'].str.lower()
    content=','.join(list(df['text'].values))
    
    wordcloud=WordCloud(background_color="white", max_words=5000, contour_width=3, 
                        contour_color='steelblue')
    wordcloud.generate(content)
    wordcloud.to_file('first_wordcloud.png')

def main():
    read_data(args.address)

if __name__ == "__main__":
    main()