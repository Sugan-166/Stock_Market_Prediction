"""
Twitter Data Scraper

This script connects to the Twitter API to scrape tweets based on specific keywords.
The collected data is stored in a CSV file for further processing.

Dependencies:
- tweepy: For accessing the Twitter API.
- pandas: For managing data.
- python-dotenv: For managing sensitive credentials in a .env file.

Setup:
1. Create a `.env` file in the same directory with the following variables:
   - TWITTER_API_KEY
   - TWITTER_API_SECRET
   - TWITTER_ACCESS_TOKEN
   - TWITTER_ACCESS_SECRET
2. Install dependencies using the provided `requirements.txt`.

"""

import tweepy
import pandas as pd
import os
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()
API_KEY = os.getenv("TWITTER_API_KEY")
API_SECRET = os.getenv("TWITTER_API_SECRET")
ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
api = tweepy.API(auth)

def scrape_tweets(keyword, max_tweets=100):
    """
    Scrape tweets based on a keyword.

    Args:
        keyword (str): The keyword to search for.
        max_tweets (int): The maximum number of tweets to scrape.

    Returns:
        pd.DataFrame: A dataframe containing tweet details.
    """
    tweets = []
    for tweet in tweepy.Cursor(api.search_tweets, q=keyword, lang="en", tweet_mode="extended").items(max_tweets):
        tweets.append({
            "created_at": tweet.created_at,
            "text": tweet.full_text,
            "retweets": tweet.retweet_count,
            "likes": tweet.favorite_count
        })
    return pd.DataFrame(tweets)

if __name__ == "__main__":
    keyword = "#StockMarket"
    data = scrape_tweets(keyword, max_tweets=500)
    data.to_csv("data/raw_data.csv", index=False)
    print("Scraped tweets saved to data/raw_data.csv")
