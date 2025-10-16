https://kgptalkie.com/real-time-twitter-sentiment-analysis

# Real-Time Twitter Sentiment Analysis

Published by [georgiannacambel](https://kgptalkie.com/real-time-twitter-sentiment-analysis) on 2 September 2020

## Donald Trump vs Warren Twitter Sentiment | US Election 2020

In this project, we are going to extract live data from Twitter related to **Donald Trump** and **Elizabeth Warren**. We are going to analyze the sentiment of the data and then plot the data in a single graph which will update in real time.

### Prerequisites

Before starting this project, you will have to create a new application by visiting this link:  
https://developer.twitter.com/app/new

Here, you will have to create a developer account. After you sign up for the developer account, your Twitter profile will be reviewed. You will get an email similar to this.

After that, you can create a new application. You will have to give details about the project you are using the API for. After you create the application, you will get 4 keys:

- **API key**
- **API secret key**
- **Access token**
- **Access token secret**

### Installing and Importing Libraries

Install `tweepy` to get data from Twitter:

```python
!pip install tweepy
```

Import `tweepy`:

```python
import tweepy
```

### Authorization

```python
consumer_key = "SaMSYfPnpQXsgeEFywA8pLg1C"
consumer_secret = "o9BjVfJHhxWmPOAT39f7i0KHJuwGb8r9k1VjHQvl4q51Gaz5I5"
access_token = "2238923408-XqfhQ40evLEZSVDUYAlZmLJF6IJXR0SCtp04Xy7"
access_token_secret = "BKqoLg2YglIqm0txjSI9oahhecxWox6a4q7g66iQe5ZvA"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
```

### Getting Twitter Timeline

```python
twittes = api.home_timeline()
for tweet in twittes:
    print(tweet.text)
```

### Streaming and Sentiment Analysis

Import necessary libraries:

```python
from tweepy import Stream, StreamListener
import json
from textblob import TextBlob
import re
import csv
```

Create a CSV file to save data:

```python
header_name = ['Trump', 'Warren']
with open('sentiment.csv', 'w') as file:
    writer = csv.DictWriter(file, fieldnames=header_name)
    writer.writeheader()
```

Define the `Listener` class:

```python
class Listener(StreamListener):
    def on_data(self, data):
        raw_twitts = json.loads(data)
        try:
            tweets = raw_twitts['text']
            tweets = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z\t])|(\w+:\/\/\S+)", " ", tweets).split())
            tweets = ' '.join(re.sub('RT', ' ', tweets).split())
            
            blob = TextBlob(tweets.strip())
            global trump, warren
            
            trump_sentiment = 0
            warren_sentiment = 0
            
            for sent in blob.sentences:
                if "Trump" in sent and "Warren" not in sent:
                    trump_sentiment = trump_sentiment + sent.sentiment.polarity
                else:
                    warren_sentiment = warren_sentiment + sent.sentiment.polarity
            
            trump = trump + trump_sentiment
            warren = warren + warren_sentiment
            
            with open('sentiment.csv', 'a') as file:
                writer = csv.DictWriter(file, fieldnames=header_name)
                info = {
                    'Trump': trump,
                    'Warren': warren
                }
                writer.writerow(info)
            
            print(tweets)
            print()
        except:
            print('Found an Error')
    
    def on_error(self, status):
        print(status)
```

Stream tweets containing "Trump" or "Warren":

```python
twitter_stream = Stream(auth, Listener())
twitter_stream.filter(track=['Trump', 'Warren'])
```

### Example Tweets

- **Source:** https://kgptalkie.com/real-time-twitter-sentiment-analysis  
  *Honored to be quoted in this piece along with amp Agree with Tribe s conclusion that obstructing*

- **Source:** https://kgptalkie.com/real-time-twitter-sentiment-analysis  
  *BREAKING New York Gov Andrew Cuomo just responded to Trump switching residency to Florida HE SAID THIS Good rid*

- **Source:** https://kgptalkie.com/real-time-twitter-sentiment-analysis  
  *I suspected but President Trump showed us how deep and dangerous it is*

- **Source:** https://kgptalkie.com/real-time-twitter-sentiment-analysis  
  *yall mad at yg for kicking a grown ass man off his stage at HIS event but yall won t be mad at donald trump for kicking*

- **Source:** https://kgptalkie.com/real-time-twitter-sentiment-analysis  
  *I see many long breadlines in the future under a President Warren*

- **Source:** https://kgptalkie.com/real-time-twitter-sentiment-analysis  
  *The greatest traitors to the Ummah in our time are the Saudi Regime and those who support it What other regime in thi*

- **Source:** https://kgptalkie.com/real-time-twitter-sentiment-analysis  
  *It has been 3 years and Donald Trump hasn t done anything wrong Donald Trump hasn t done a single thing of which he*

### Plotting the Data

Import necessary libraries:

```python
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
%matplotlib notebook
```

Set the style and frame length:

```python
plt.style.use('fivethirtyeight')
frame_len = 10000
fig = plt.figure(figsize=(9, 6))
```

Define the `animate` function:

```python
def animate(i):
    data = pd.read_csv('sentiment.csv')
    y1 = data['Trump']
    y2 = data['Warren']
    
    if len(y1) <= frame_len:
        plt.cla()
        plt.plot(y1, label='Donald Trump')
        plt.plot(y2, label='Elizabeth Warren')
    else:
        plt.cla()
        plt.plot(y1[-frame_len:], label='Donald Trump')
        plt.plot(y2[-frame_len:], label='Elizabeth Warren')
    
    plt.legend(loc='upper left')
    plt.tight_layout()
```

Create the animation:

```python
ani = FuncAnimation(plt.gcf(), animate, interval=1000)
```

## Summary

We have seen many things in this project. We started by getting access to the keys. Then we authorized our API using the keys. After that, we scrapped live filtered tweets from Twitter. We pre-processed them, found their sentiment, and stored the sentiment in a CSV file in a proper format. Then we read the CSV file and created a function that continuously plots the data. Further on, you can change the keywords and filter tweets according to your need and draw various plots.