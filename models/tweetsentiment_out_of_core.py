# logistic regression model for online learning
import numpy as np
import re
from nltk.corpus import stopwords
stop = stopwords.words('english')

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) +        ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

# utility functions for getting tweets in batches for training
def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)  # skipping the header
        for line in csv:
            splitted_line = line.split(",")
            label = int(splitted_line[1])
            text = ",".join(splitted_line[3:]).strip()
            yield text, label

def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y
    
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)
clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
doc_stream = stream_docs(path='/home/roope/projects/sort-by-sentiment/training-data/Sentiment Analysis Dataset.csv')

import pyprind
pbar = pyprind.ProgBar(45)
classes = np.array([0,1])
for _ in range(1300):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes = classes)
    pbar.update()
    
X_test, y_test = get_minibatch(doc_stream, size = 5000)
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test)) # accuracy = 0.78! for the test set. 


# saving clf classifier for future use. making folder and serializing via pickle 
import pickle
import os 
dest = os.path.join('tweetclassifier_outofcore', 'pickled_objects')
if not os.path.exists(dest):
    os.makedirs(dest)

pickle.dump(stop,
            open(os.path.join(dest, 'stopwords.pkl'), 'wb'),
            protocol=4)

pickle.dump(clf,
            open(os.path.join(dest, 'classifier.pkl'),'wb'),
            protocol=4) 