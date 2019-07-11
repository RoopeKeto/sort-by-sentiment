#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import LatentDirichletAllocation
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version


# # # GET AND CLEAN THE DATA # # #

# the data has Sentiment column containing labels 1 = positive sentiment, 0 = negative
data = pd.read_csv('/home/roope/projects/sort-by-sentiment/training-data/Sentiment Analysis Dataset.csv',
                   error_bad_lines=False)

# removing whitespace 
data['SentimentText'] = data['SentimentText'].apply(str.strip)

# shuffling the data
data = data.reindex(np.random.permutation(data.index))

# using utility function to deal with emoticons and non words
def preprocessor(text):
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text

data['SentimentText'] = data['SentimentText'].apply(preprocessor)

# tokenizers
def tokenizer(text):
    return text.split()

porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

# getting stopwords
nltk.download('stopwords')
stop = stopwords.words('english')

# # # Building logistic regression model # # # 
    
# splitting data into train and test sets 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['SentimentText'], data['Sentiment'])
#splitting to smaller sizes (for faster testing)
#X_train = X_train[0:2500]
#y_train = y_train[0:2500]

# using tfidf
tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

# using grid search
# param_grid = [{'vect__ngram_range': [(1, 1)],
#                'vect__stop_words': [stop, None],
#                'vect__tokenizer': [tokenizer, tokenizer_porter],
#                'clf__penalty': ['l1', 'l2'],
#                'clf__C': [1.0, 10.0, 100.0]},
#               {'vect__ngram_range': [(1, 1)],
#                'vect__stop_words': [stop, None],
#                'vect__tokenizer': [tokenizer, tokenizer_porter],
#                'vect__use_idf':[False],
#                'vect__norm':[None],
#                'clf__penalty': ['l1', 'l2'],
#                'clf__C': [1.0, 10.0, 100.0]},
#               ]

# using following for performance reasons (full data)
param_grid = [{'vect__ngram_range': [(1, 1)],
                   'vect__stop_words': [None],
                   'vect__tokenizer': [tokenizer],
                    'clf__penalty': ['l2'],
                    'clf__C': [10.0]},
                   ]


lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=-1)

gs_lr_tfidf.fit(X_train, y_train)

print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)


clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test)) # about 80 % was the accuracy
# check the results

