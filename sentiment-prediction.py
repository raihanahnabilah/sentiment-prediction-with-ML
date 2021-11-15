#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 13:44:00 2021
@author: hanabilaf
"""

####################################################
# METHOD 3: MAIN METHOD
# EXPLANATIONS ARE PROVIDED IN THE PDF REPORT
####################################################

import nltk
from nltk.corpus import movie_reviews
import os
import pandas as pd

# Creating the dataframe and processing the text
def load_reviews():
    keys = ['filename','kind','text']
    res = {}
    for i in keys:
        res[i] = []
    neg_files = movie_reviews.fileids('neg')
    for file in neg_files:
        res['filename'].append(os.path.basename(file))
        res['kind'].append('neg')
        res['text'].append(movie_reviews.raw(file))
    pos_files = movie_reviews.fileids('pos')
    for file in pos_files:
        res['filename'].append(os.path.basename(file))
        res['kind'].append('pos')
        res['text'].append(movie_reviews.raw(file))
    return pd.DataFrame.from_dict(res)

data = load_reviews()

# Tokenization, filtering, and lemmatize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
stopWords = set(stopwords.words("english"))
import nltk.stem  as stem
wordnet_lemmatizer = stem.WordNetLemmatizer()

def tokenize_filter_and_lemmatize(text):
    word = word_tokenize(text)
    filter_1 = [w for w in word if w.isalpha()]
    filter_2 = [w for w in filter_1 if w not in stopWords]
    lemmatize = [wordnet_lemmatizer.lemmatize(w) for w in filter_2]
    return lemmatize

data['text'] = data['text'].apply(lambda x: tokenize_filter_and_lemmatize(x))

data['text'] = data.text.apply(lambda x: ' '.join(x))

# Data training
X = data.text.to_numpy()
Y = data.kind.to_numpy()

from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
X = vec.fit_transform(X)

# Building the model
from sklearn.svm import LinearSVC

LinearSVC = LinearSVC()
LinearSVC.fit(X, Y)

# Predict sentiment
def predict_sentiment(text):
    words = tokenize_filter_and_lemmatize(text)
    review = [" ".join(words)]
    test = vec.transform(review)
    predict = LinearSVC.predict(test)[0]
    return predict

####################################################
# USING VOTE RESULT CLASS BY COMBINING ALL MODELS
# COMMENTARY PROVIDED IN THE REPORT
####################################################

# from sklearn.naive_bayes import MultinomialNB, BernoulliNB
# from sklearn.linear_model import LogisticRegression, SGDClassifier
# from sklearn.svm import SVC, LinearSVC, NuSVC

# MultinomialNB = MultinomialNB()
# MultinomialNB.fit(X, Y)

# BernoulliNB = BernoulliNB()
# BernoulliNB.fit(X, Y)

# SGD = SGDClassifier()
# SGD.fit(X, Y)

# LogisticRegression = LogisticRegression()
# LogisticRegression.fit(X,Y)

# SVC = SVC()
# SVC.fit(X,Y)

# NuSVC = NuSVC()
# NuSVC.fit(X,Y)

# from nltk.classify import ClassifierI
# from statistics import mode

# class VoteResult(ClassifierI):
#     def __init__(self, *classifiers):
#         self._classifiers = classifiers

#     def classify(self,test_set):
#     votes = []
#     for c in self._classifiers:
#         v = c.predict(test_set)[0]
#         votes.append(v)
#     return mode(votes)

# vote_result = VoteResult(SGD,
#                          MultinomialNB,
#                          BernoulliNB,
#                          LogisticRegression,
#                          SVC,
#                          LinearSVC,
#                          NuSVC)

# def predict_sentiment(text):
#     words = tokenize_filter_and_lemmatize(text)
#     review = [" ".join(words)]
#     test = vec.transform(review)
#     predict = vote_result.classify(test)
#     return predict