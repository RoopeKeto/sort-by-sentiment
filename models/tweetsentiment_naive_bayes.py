import pandas as pd 
import numpy as np
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

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


# # # BUILDING MODEL # # #

# splitting data into train and test sets 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['SentimentText'], data['Sentiment'])

# bags of words representation
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(stop_words='english')

X_train_cv = count_vect.fit_transform(X_train)
X_test_cv = count_vect.transform(X_test)


# fitting the multinomial naive bayes model and making predictions
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_cv, y_train)
predictions = naive_bayes.predict(X_test_cv)

# check the results
print('Accuracy score: ', accuracy_score(y_test, predictions)) # 0.77
print('Precision score: ', precision_score(y_test, predictions)) # 0.78
print('Recall score: ', recall_score(y_test, predictions)) # 0.74

