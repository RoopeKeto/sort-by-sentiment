from flask import Flask, render_template
import sqlite3
from flask import g
import os

DATABASE = 'database/twitter.db'

def get_db():
    db = getattr(g, '_database', None)

# dummy data for testing purposes
tweets = [
    {   
        'username': 'Dummy_negative',
        'posted_date': '1.1.2019',
        'tweet_text': "This sucks really bad. I don't like this at all",
        'sentiment': 0
    },
    {
        'username': 'Dummy_positive',
        'posted_date': '1.2.2019',
        'tweet_text': "Life is good. I'm in love. Great job!",
        'sentiment': 1
    },
    {
        'username': 'Dummy_middle',
        'posted_date': '1.3.2019',
        'tweet_text': 'It is okey, I would say',
        'sentiment': 0.55
    }
]

app = Flask(__name__)

@app.route('/')
def index():
    sorted_tweets = sorted(tweets, key = lambda i: i['sentiment'], reverse=True)
    return render_template('sort_by_sentiment_app.html', tweets=sorted_tweets)


if __name__ == '__main__':
    app.run(debug=True, host=os.getenv('IP', '0.0.0.0'), 
            port=int(os.getenv('PORT', 4444)))