# -*- coding: utf-8 -*-
"""
@author: zding
"""

from collections import Counter
import preprocessing
import re
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional, SpatialDropout1D
from keras.preprocessing import text, sequence
from keras.initializers import Constant
from keras.optimizers import Adam

args = argparse.ArgumentParser(description='Program description.')
args.add_argument('-a','--address', default='train.csv', help='the address of the file to be process')
args.add_argument('-t','--test', default='test.csv', help='the address of the test file')
args.add_argument('-e', '--embedding_size', default=100, type=int, help='Embedding dimension size')
args.add_argument('-nf', '--number_features', default=15000, type=int, help='max length of features')
args.add_argument('-g', '--glove_file', default='glove.twitter.27B\glove.twitter.27B.200d.txt', type=str, help="glove document address")
args.add_argument('-d', '--dropout', default=0.25, type=float)
args.add_argument('-l', '--learning_rate', default=0.0005, type=float)
args.add_argument('-ep', '--epoch', default=16, type=int)
args = args.parse_args()

df=preprocessing.as_list(args.address)
test_df=preprocessing.as_list(args.test)
disaster_unigram_dict=Counter()
nondisaster_unigram_dict=Counter()

def feature_engineering(df):
    """
    getting features from the existing attributes from the dataframe
    """
    #word count
    df['word_count']=df['text_list'].apply(lambda x: len(x))
    #unique words
    df['unique_word']=df['text_list'].apply(lambda x: len(set(x)))
    #url
    df['url']=df['text'].apply(lambda x: 
        len(re.findall('(https?|ftp)://', x)))
    #hashtag
    df['hashtag']=df['text'].apply(lambda x: len([t for t in str(x) if t=='#']))
    #at
    df['at']=df['text'].apply(lambda x: len([t for t in str(x) if t=='@']))
    return df

def cleaning_for_glove(df):
    '''
    This function does extra cleaning for the glove model, including removing url,
    hashtags, at tag
    '''
    text_df=df['text_list'].copy()
    labels=df['target'].tolist()
    new_text=[]
    new_data=[]
    for index, l in text_df.items():
        new_list=[]
        for item in l:
            #url
            if not item.startswith('http') and \
            not item.startswith('#') and \
            not item.startswith('@'):
                new_list.append(item)
        new_text.append(new_list)
    for l in new_text:
        new_data.append(' '.join(l))
    glove_df = pd.DataFrame({'text': new_data, 'target': labels})
    return glove_df

def construct_df(df):
    """
    get the dataframe ready to train
    """
    texts=df['text_list'].values
    labels=df['target'].tolist()
    new_text=[]
    for l in texts:
        new_text.append(' '.join(l))
    new_df = pd.DataFrame({'text': new_text, 'target': labels})
    return new_df

def test_glove(df):
    text_df=df['text_list'].copy()
    new_text=[]
    new_data=[]
    for index, l in text_df.items():
        new_list=[]
        for item in l:
            #url
            if not item.startswith('http') and \
            not item.startswith('#') and \
            not item.startswith('@'):
                new_list.append(item)
        new_text.append(new_list)
    for l in new_text:
        new_data.append(' '.join(l))
    glove_df = pd.DataFrame({'text': new_data})
    return glove_df



def generate_dicts():
    #unigram
    for index, row in df.iterrows():
        if(row['target']==1):
            for token in row['text_list']:
                disaster_unigram_dict[token]+=1
        else:
            for token in row['text_list']:
                nondisaster_unigram_dict+=1
                
def glove_method(train_df, test_df):
    """
    This function does the golve embedding classification
    """
    tokenizer=text.Tokenizer(num_words=args.number_features)
    tokenizer.fit_on_texts(train_df.text)
    token_train=tokenizer.texts_to_sequences(train_df.text)
    x=sequence.pad_sequences(token_train, maxlen=50)
    token_test=tokenizer.texts_to_sequences(test_df.text)
    x_t=sequence.pad_sequences(token_test, maxlen=50)
    
    embedding_metrix={}
    with open(args.glove_file, 'r', encoding='utf8') as f:
        for line in f:
            v=line.split()
            w=v[0]
            vec=np.asarray(v[1:], 'float32')
            embedding_metrix[w]=vec
            
    n_words=len(tokenizer.word_index)+1
    embedding_matrix=np.zeros((n_words, 200))
    
    for word, i in tokenizer.word_index.items():
        if i<n_words:
            embedding_vec=embedding_metrix.get(word)
            if embedding_vec is not None:
                embedding_matrix[i]=embedding_vec
    print(x[0][0:])
    
    model=Sequential()
    embedding_layer=Embedding(n_words,
                              200,
                              embeddings_initializer=Constant(embedding_matrix),
                              input_length=50,
                              trainable=False)
    model.add(embedding_layer)
    model.add(SpatialDropout1D(args.dropout))
    model.add(Bidirectional(LSTM(200, dropout=args.dropout, recurrent_dropout=args.dropout)))
    model.add(Dense(125, activation='sigmoid'))
    model.add(Dropout(args.dropout))
    #model.add(LSTM(100, dropout=args.dropout))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=args.learning_rate), metrics=['accuracy'])
    #shuffle set to false here for the error analysis
    x_train, x_test, y_train, y_test=train_test_split(x, train_df['target'].values, test_size=0.2)
    
    model_history=model.fit(x_train, y_train, batch_size=4, epochs=args.epoch, validation_data=(x_test, y_test))
    y_pred = model.predict(x_test)
    return y_pred
    
def machine_learnings(train_df):
    """
    This function implement the Naive Bayes classification for this project
    """
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    x=vectorizer.fit_transform(train_df['text'].values)
    y=train_df['target'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    model=MultinomialNB().fit(x_train, y_train)
    prediction=model.predict(x_test)
    
    #print results
    print(classification_report(y_test, prediction))
    
def error_analysis(df, pred_result):
    """
    error analysis: extract all the tweets that are mislabeled by the glove model
    in the test dataset to a csv file
    """
    error_text=[]
    for i in range(len(pred_result)):
        if pred_result[i][0]<.5:
            error_text.append(df.iloc[[-762+i]]['text'].values)
    err_df=pd.DataFrame(error_text)
    err_df.to_csv("error_text.csv", mode='w')
    

def main():
    df=preprocessing.as_list(args.address)
    #baseline
    train_df=construct_df(df)
    machine_learnings(glove_train_df)
    #glove lstm model
    train_df=construct_df(df)
    glove_train_df=cleaning_for_glove(df)
    glove_test_df=test_glove(test_df)
    pred_result=glove_method(glove_train_df, glove_test_df)
    error_analysis(df, pred_result)
    
    
if __name__=="__main__":
    main()
