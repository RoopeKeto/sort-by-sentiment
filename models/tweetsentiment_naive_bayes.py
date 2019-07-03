import pandas as pd 
import numpy as np

# # # GET AND CLEAN THE DATA # # #
data = pd.read_csv('/home/roope/projects/sort-by-sentiment/training-data/Sentiment Analysis Dataset.csv',
                   error_bad_lines=False)

data['SentimentText'] = data['SentimentText'].apply(str.strip)

# # # Using Bag of words method # # #
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform()


