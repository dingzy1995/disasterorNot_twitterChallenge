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
import preprocessing
import pandas as pd
import re
import argparse
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

args = argparse.ArgumentParser(description='Program description.')
args.add_argument('-a','--address', default='train.csv', help='the address of the file to be process')
args = args.parse_args()




def read_data(address):
    df=preprocessing.as_list(address)
    #only need the cleaned tweet content for this task
    df.drop(columns=['id', 'keyword', 'location', 'text'], axis=1)
    disaster_content=[]
    nondisaster_content=[]
    for index, row in df.iterrows():
        temp=' '.join(row['text_list'])
        if row['target']==1:
            disaster_content.append(temp)
        else:
            nondisaster_content.append(temp)
    #tokenize the tweets
    disaster_token = word_tokenize(','.join(disaster_content))
    nondisaster_token = word_tokenize(','.join(nondisaster_content))
    disaster_tweet=','.join(disaster_token)
    nondisaster_tweet=','.join(nondisaster_token)
    wordcloud=WordCloud(background_color="white", max_words=5000, contour_width=3, 
                        contour_color='steelblue')
    wordcloud.generate(disaster_tweet)
    wordcloud.to_file('disasterwordcloud.png')
    wordcloud.generate(nondisaster_tweet)
    wordcloud.to_file('nondisasterwordcloud.png')

def main():
    read_data(args.address)

if __name__ == "__main__":
    main()