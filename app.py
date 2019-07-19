from flask import Flask, render_template, url_for, redirect
import os
import numpy as np
import pickle 
from vectorizer import vect
from twitter_search import search_tweets
from forms import SearchForm
import wordcloud
from flask_cachebuster import CacheBuster
import pygal 

app = Flask(__name__)
app.config['SECRET_KEY'] = '0b248914a4417846b62d195c17626830'

# helper function for getting tweets by searchword and quantity specification
def get_tweets(search_word, number_of_tweets):
    raw_tweets = search_tweets(search_word, number_of_tweets)
    text = ""
    tweets = []

    for tweet in raw_tweets:
        tweet_dict = {}
        if hasattr(tweet, 'text'):
            tweet_dict['tweet_text'] = tweet.text
            text = text + " " + tweet.text
        else:
            tweet_dict['tweet_text'] = tweet.full_text
            text = text + " " + tweet.full_text
        
        tweet_dict['username'] = tweet.author.name
        tweet_dict['posted_date'] = tweet.created_at
        tweet_dict['img_source'] = tweet.user.profile_image_url
        tweets.append(tweet_dict)
    return tweets, text

# initializin cachebuster
config = { 'extensions': ['.png'], 'hash_size': 5 }
cache_buster = CacheBuster(config=config)
cache_buster.init_app(app)

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
    try:
        form = SearchForm()

        if form.validate_on_submit():
            
            tweets, text_for_cloud = get_tweets(form.searchword.data, form.number_of_results.data)

            # creating wordcloud and storing to static
            stop_words = ["https", "co", "RT"] + list(wordcloud.STOPWORDS)
            word_cloud = wordcloud.WordCloud(stopwords=stop_words, background_color="rgba(255, 255, 255, 0)", mode="RGBA")
            word_cloud.generate(text_for_cloud)
            word_cloud.to_file("static/word_cloud.png")

            # calculating sentiments for tweets and adding these to dictionaries, also adding counts to bin
            bin_counts = [0,0,0,0,0,0,0,0,0,0]
            for i in range(len(tweets)):
                tweets[i]['sentiment'] = classify(tweets[i]['tweet_text'])
                if tweets[i]['sentiment'] < 0.1:
                    bin_counts[0] += 1
                elif tweets[i]['sentiment'] < 0.2:
                    bin_counts[1] += 1
                elif tweets[i]['sentiment'] < 0.3:
                    bin_counts[2] += 1
                elif tweets[i]['sentiment'] < 0.4:
                    bin_counts[3] += 1
                elif tweets[i]['sentiment'] < 0.5:
                    bin_counts[4] += 1
                elif tweets[i]['sentiment'] < 0.6:
                    bin_counts[5] += 1
                elif tweets[i]['sentiment'] < 0.7:
                    bin_counts[6] += 1
                elif tweets[i]['sentiment'] < 0.8:
                    bin_counts[7] += 1
                elif tweets[i]['sentiment'] < 0.9:
                    bin_counts[8] += 1
                else:
                    bin_counts[9] += 1

            sorted_tweets = sorted(tweets, key = lambda i: i['sentiment'], reverse=True)

            # creating histogram
            from pygal.style import Style
            custom_style = Style(
            background='transparent',
            plot_background='transparent',
            colors=('#79bdd8','#79bdd8'))
            graph = pygal.Histogram(show_legend=False, title=u'Distribution of the sentiment predictions', 
                                    x_title='0 = most negative  1 = most positive', style=custom_style)
            graph.add('Narrow bars',  [(bin_counts[0], 0, 0.1), (bin_counts[1], 0.1, 0.2), (bin_counts[2], 0.2, 0.3),
                                        (bin_counts[3], 0.3, 0.4),(bin_counts[4], 0.4, 0.5),(bin_counts[5], 0.5, 0.6),
                                        (bin_counts[6], 0.6, 0.7),(bin_counts[7], 0.7, 0.8),(bin_counts[8], 0.8, 0.9),(bin_counts[9], 0.9, 1)])
            graph_data = graph.render_data_uri()
        
            return render_template('sort_by_sentiment_app.html', tweets=sorted_tweets, form=form, graph_data=graph_data)

        return render_template('sort_by_sentiment_app.html', form=form)
    except Exception as e:
        return(str(e))

if __name__ == '__main__':
    app.run(host=os.getenv('IP', '0.0.0.0'), 
            port=int(os.getenv('PORT', 4447)))
