from __future__ import print_function
import sys
from flask import Flask, render_template, url_for, redirect
import sqlite3
from sqlite3 import Error
import os
import numpy as np
import pickle 
from twitter_search import search_tweets
from forms import SearchForm


# function for getting tweets by searchword and quantity specification

def get_tweets(search_word, number_of_tweets):
    raw_tweets = search_tweets(search_word, number_of_tweets)
    tweets = []
    for tweet in raw_tweets:
        tweet_dict = {}
        if hasattr(tweet, 'text'):
            tweet_dict['tweet_text'] = tweet.text
        else:
            tweet_dict['tweet_text'] = tweet.full_text
        
        tweet_dict['username'] = tweet.author.name
        tweet_dict['posted_date'] = tweet.created_at
        tweets.append(tweet_dict)
    return tweets

# importing hashingvectorizer    

from vectorizer import vect

# path to database
DATABASE = 'database/twitter.db'

# dummy data for testing purposes
dummydata = [
    {   
        'username': 'Dummy_negative',
        'posted_date': '1.1.2019',
        'tweet_text': "This sucks really bad. I don't like this at all",
    },
    {    'username': 'Dummy_positive',
        'posted_date': '1.2.2019',
        'tweet_text': "Life is good. I'm in love. Great job!",
    },
    {
        'username': 'Dummy_middle',
        'posted_date': '1.3.2019',
        'tweet_text': 'It is okey, I would say',
    },
    {
        'username': 'Quite_negative',
        'posted_date': '1.3.2019',
        'tweet_text': "This is lame."
    }
]

# getting data via stream API


app = Flask(__name__)

app.config['SECRET_KEY'] = '0b248914a4417846b62d195c17626830'

# setting up the classifier
cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir,
                   'tweetclassifier_outofcore',
                   'pickled_objects',
                   'classifier.pkl'
                   ), 'rb'))

def classify(tweet):
    label = {0: 'negative', 1: 'positive'}
    X = vect.transform([tweet])
    proba_y = clf.predict_proba(X)[0][1]
    return proba_y


@app.route('/', methods=['GET', 'POST'])
def index():
    # create a database connection
    try:
        conn = sqlite3.connect(DATABASE)
    except Error as e:
        print(e)

    # Getting data from form
    form = SearchForm()

    if form.validate_on_submit():
        tweets = get_tweets(form.searchword.data, form.number_of_results.data)
        print(len(tweets))
        # calculating sentiments for tweets and adding these to dictionaries
        for i in range(len(tweets)):
            tweets[i]['sentiment'] = classify(tweets[i]['tweet_text'])

        sorted_tweets = sorted(tweets, key = lambda i: i['sentiment'], reverse=True)
    
        return render_template('sort_by_sentiment_app.html', tweets=sorted_tweets, form=form)

    return render_template('sort_by_sentiment_app.html', form=form)

if __name__ == '__main__':
    app.run(debug=True, host=os.getenv('IP', '0.0.0.0'), 
            port=int(os.getenv('PORT', 4446)))

