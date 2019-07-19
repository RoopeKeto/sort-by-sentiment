#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:00:30 2019

@author: roope
"""

import tweepy
import json
import twitter_credentials

# twitter authentication and api initialization
global api
access_token = twitter_credentials.ACCESS_TOKEN 
access_token_secret = twitter_credentials.ACCESS_TOKEN_SECRET
consumer_key = twitter_credentials.CONSUMER_KEY
consumer_secret = twitter_credentials.CONSUMER_SECRET
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

# # # utility functions for displaying and searching tweets
def print_tweet(tweet):
    '''
    Prints out short info on tweet and the extended tweet text
    '''
    print('\n')
    print(f'{tweet.user.screen_name}, {tweet.user.name}, {tweet.created_at}')
    if hasattr(tweet, 'text'):
        print(tweet.text)
    else:
        print(tweet.full_text)


def search_tweets(search_word, tweet_quantity, ignore_retweets=True, language='en'):
    ''' returns tweets that contain argument "search_word" with quantity of "tweet_quantity" '''
    tweets = []
    for tweet in tweepy.Cursor(api.search, q=search_word, lang=language, tweet_mode="extended").items(tweet_quantity):
        if ignore_retweets:
            if not hasattr(tweet, 'retweeted_status'):
                tweets.append(tweet)
        else:
            tweets.append(tweet)
    return tweets

def get_tweet_text(tweet):
    
    if hasattr(tweet, 'text'):
        return tweet.text
    else:
        return tweet.full_text
