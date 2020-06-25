# In[1]:


#listing twitter json files
import glob
import pandas as pd
import numpy as np
import json
import re
import string
import nltk
import itertools
import collections
import matplotlib.pyplot as plt


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


# In[2]:


## create pandas DataFrame for further analysis
df_tweet = pd.DataFrame(tweets)
df_tweet.tail()


# In[3]:


#Analysis of tweets

#proportion of tweet languages
ang_counts = Counter(df_tweet['lang'])
lang_counts = pd.DataFrame(lang_counts.most_common(), columns = ['Language', 'Count'])
lang_counts['Proportion'] = round((lang_counts['Count']/sum(lang_counts['Count']))*100,2)
lang_counts


# In[7]:


#filtering english tweets
df_tweet_en = df_tweet[df_tweet['lang']=='en']

#resetting dataframe index to datetime column
df_tweet_en['created_at'] = pd.to_datetime(df_tweet_en['created_at'])
df_tweet_en = df_tweet_en.set_index('created_at')


# In[8]:


#Number and proportion of English tweets captured in a given period of time
#total number of tweets
total_tweets = len(df_tweet_en)

#proportion of English tweets as compared to other languages
prop = (len(df_tweet_en)/len(df_tweet))*100

#total duration (in minutes) during which twitter activity captured
delta = max(df_tweet_en.index) - min(df_tweet_en.index)
d_minutes = (delta.seconds % 3600) // 60

print("There are a total of {} tweets in the English language, which make up {}% of the total tweets captured in {} minutes.".format(len(df_tweet_en), round(prop,2), d_minutes))
# print("There are a total of {} tweets in English.".format(len(df_tweet_en)))


# In[11]:


#combining all tweets into one blob
text = " ".join(i for i in df_tweet_en.text)

text = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', text)

from nltk.corpus import stopwords
from tokenize import tokenize
from nltk.tokenize import RegexpTokenizer
nltk.download('stopwords')
tokenizer = RegexpTokenizer(r'\w+')

stopwords = nltk.corpus.stopwords.words('english')
newStopWords = ['http','rt', 'https', 'co', 'coronavirus']
for i in newStopWords:
    stopwords.append(i)
    
text1 = df_tweet_en.apply(lambda row: tokenizer.tokenize(row['text']), axis=1)
text2 = list(itertools.chain(*text1))
text3 = [tweet.strip(' ').lower() for tweet in text2]        
text4 = [tweet for tweet in text3 if tweet not in stopwords]


# In[13]:


#Create counter
text6 = []
for i in text4:
    word_cleaned = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', i)
    text6.append(word_cleaned)

counts = collections.Counter(text6)
top20 = pd.DataFrame(counts.most_common(20), columns = ['Word', 'Count']) 
text7 = " ".join(i for i in text6)

top20


# In[16]:


#Visualizing tweets using a wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
get_ipython().run_line_magic('matplotlib', 'inline')

# Generate a word cloud image
wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white").generate(text7)
from matplotlib.pyplot import figure
figure(num=None, figsize=(10, 7.5), dpi=80, facecolor='w', edgecolor='k')
# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[17]:


# Load SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Instantiate new SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Generate sentiment scores
sentiment_scores = df_tweet_en['text'].apply(sid.polarity_scores)
sentiment = sentiment_scores.apply(lambda x: x['compound'])
sentiment = sentiment.resample('1T', axis=0).mean()


# In[18]:


#Capturing minute by minute sentiment variation in tweets
from datetime import timedelta
sentiment.index = pd.to_timedelta((sentiment.index.strftime('%H:%M:%S')))
minutes = (sentiment.index.seconds % 3600) // 60
delta_min = range(len(minutes))


# In[23]:


#Plotting minute by minute sentiment variation in tweets
figure(num=None, figsize=(12, 6))
plt.plot(delta_min, sentiment, color = 'red', label='sentiment')
plt.axhline(y=0, color='b', linestyle=':')
plt.xlabel('Minute')
plt.ylabel('Sentiment')
plt.title(' "#coronavirus" tweet sentiments across time')
plt.xticks(rotation=90)
plt.xticks(np.arange(0,len(delta_min), step=1))
plt.show()





