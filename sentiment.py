# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 10:50:17 2018

@author: WenDong Zheng
"""

import os
import re
import warnings
warnings.simplefilter("ignore", UserWarning)
from matplotlib import pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from string import punctuation
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, roc_auc_score
from sklearn.externals import joblib
import scipy
from scipy.sparse import hstack

data = pd.read_csv('Sentiment Analysis Dataset.csv', encoding='latin1', usecols=['Sentiment', 'SentimentText'])
data.columns = ['sentiment', 'text']
data = data.sample(frac=1, random_state=42)
print(data.shape)
for row in data.head(10).iterrows():
    print(row[1]['sentiment'], row[1]['text'])
    
def tokenize(tweet):
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r"#(\w+)", '', tweet)
    tweet = re.sub(r"@(\w+)", '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = tweet.strip().lower()
    tokens = word_tokenize(tweet)
    return tokens


data['tokens'] = data.text.progress_map(tokenize)
data['cleaned_text'] = data['tokens'].map(lambda tokens: ' '.join(tokens))
data[['sentiment', 'cleaned_text']].to_csv('cleaned_text.csv')

data = pd.read_csv('cleaned_text.csv')
print(data.shape)