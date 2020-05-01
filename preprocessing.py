# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:28:32 2020

@author: zding
"""
import argparse
import string
import pandas as pd
from nltk.stem import *
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

args = argparse.ArgumentParser(description='Program description.')
args.add_argument('-a','--address', default='train.csv', help='the address of the file to be process')
args = args.parse_args()

def read_data(address):
    with open(address, 'r') as file:
        df=pd.read_csv(file)
        df_without_null=df.dropna()
        df_without_null.to_csv(address+'_without_null.csv')
        

def as_list(address):
    """
    this function returns data as list with words in them being lower cased,
    stemmed, stop words
    removed.
    """
    result=[]
    stemmer=PorterStemmer()
    tweet_punctuation=string.punctuation.replace('@', '').replace('#', '')
    with open(address, 'r') as file:
        df=pd.read_csv(file)
        df=df.dropna()
        for index, row in df.iterrows():
            text=row['text']
            text=text.lower()
            text_list=text.split()
            word_list=[]
            for token in text_list:
                token=token.lower()
                token=token.strip(tweet_punctuation)
                if token not in stop_words and len(token) > 3:
                    word_list.append(stemmer.stem(WordNetLemmatizer().lemmatize(token, pos='v')))
            result.append(word_list)
        df['text_list']=result
        print(result[0])
    return df

def main():
    as_list(args.address)

if __name__ == "__main__":
    main()