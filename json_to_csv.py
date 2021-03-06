#listing twitter json files
import glob
import pandas as pd
import numpy as np
import json

#extracting tweets
tweets = []
files  = list(glob.iglob('streamer*.json'))

#cleaning and formatting tweets
for f in files:
    fh = open(f, 'r', encoding = 'utf-8')
    tweets_json = fh.read().split("\n")

    ## remove empty lines
    tweets_json = list(filter(len, tweets_json))

    ## parse each tweet
    for tweet in tweets_json:
        try:
            tweet_obj = json.loads(tweet)

            ## flatten the file to include quoted status and retweeted status info
            if 'quoted_status' in tweet_obj:
                tweet_obj['quoted_status-text'] = tweet_obj['quoted_status']['text'] 
                tweet_obj['quoted_status-user-screen_name'] = tweet_obj['quoted_status']['user']['screen_name']
                tweet_obj['quoted_status-extended_tweet-full_text'] = tweet_obj['quoted_status']['extended_tweet']['full_text']

            if 'retweeted_status' in tweet_obj:
                tweet_obj['retweeted_status-user-screen_name'] = tweet_obj['retweeted_status']['user']['screen_name']
                tweet_obj['retweeted_status-text'] = tweet_obj['retweeted_status']['text']
                tweet_obj['retweeted_status-extended_tweet-full_text'] = tweet_obj['retweeted_status']['extended_tweet']['full_text']
            
            if 'extended_tweet' in tweet_obj:
              tweet_obj['extended_tweet-full_text'] = tweet_obj['extended_tweet']['full_text']


            tweet_obj['user-screen_name'] = tweet_obj['user']['screen_name']

            tweets.append(tweet_obj)
        except:
            pass

## create pandas DataFrame for further analysis
df_tweet = pd.DataFrame(tweets)
df_tweet.tail()

#Save dataframe as csv
df_tweet.to_csv('/User/Directory/Location/Tweets.csv')
