# -*- coding: utf-8 -*-
"""
@author: zding
"""
import emoji
import re
import argparse
import string
import pandas as pd
from nltk.stem import *
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

args = argparse.ArgumentParser(description='Program description.')
args.add_argument('-a','--address', default='train.csv', help='the address of the file to be process')
args = args.parse_args()

def explore(address):
    df=pd.read_csv(address)
    keyword_na=df['keyword'].isna().sum()
    location_na=df['location'].isna().sum()
    keyword_uni=df['keyword'].nunique()
    location_uni=df['location'].nunique()
    print(str(keyword_na)+' '+str(location_na))
    print(str(keyword_uni)+' '+str(location_uni))

def read_data(address):
    with open(address, 'r') as file:
        df=pd.read_csv(file)
        df_without_null=df.dropna()
        df_without_null.to_csv(address+'_without_null.csv')
        
def find_emoji(text):
    emo_text=emoji.demojize(text)
    line=re.findall(r'\:(.*?)\:', emo_text)
    return line

"""
emoji removing is copied from https://www.kaggle.com/raenish/cheatsheet-text-helper-functions
"""
def remove_emoji(text):
    emoji_pattern = re.compile("["
                                 u"\U0001F600-\U0001F64F"
                                 u"\U0001F300-\U0001F5FF"
                                 u"\U0001F680-\U0001F6FF"
                                 u"\U0001F1E0-\U0001F1FF"
                                 u"\U00002702-\U0001F251"
                                 "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)
        
def remove_html(text):
    html_pattern = re.compile(r'<.*?>')
    return html_pattern.sub(r'', text)

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
        df.fillna('', inplace=True)
        for index, row in df.iterrows():
            text=row['text']
            text=text.lower()
            text=remove_emoji(text)
            text=remove_html(text)
            text_list=text.split()
            word_list=[]
            for token in text_list:
                token=token.lower()
                token=token.strip(tweet_punctuation)
                if token not in stop_words and len(token) > 3:
                    word_list.append(stemmer.stem(WordNetLemmatizer().lemmatize(token, pos='v')))
            result.append(word_list)
        df['text_list']=result
    return df

def main():
    explore(args.address)

if __name__ == "__main__":
    main()