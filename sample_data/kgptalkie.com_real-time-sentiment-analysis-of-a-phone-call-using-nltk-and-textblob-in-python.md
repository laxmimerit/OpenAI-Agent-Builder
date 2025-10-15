https://kgptalkie.com/real-time-sentiment-analysis-of-a-phone-call-using-nltk-and-textblob-in-python

# Real-Time Sentiment Analysis of a Phone Call Using NLTK and TextBlob in Python

Published by **georgiannacambel** on **2 September 2020**

## Speech to Text Conversion and Real-Time Sentiment Analysis

In this project, we are going to analyse the sentiment of the call. We are first going to convert the speech to text and then analyse the sentiment using [TextBlob](https://textblob.readthedocs.io/).

### TextBlob

TextBlob is a [Python](https://www.python.org/) library for processing textual data. It provides a simple API for diving into common natural language processing (NLP) tasks such as:

- Part-of-speech tagging
- Noun phrase extraction
- Sentiment analysis
- Classification
- Translation

Install TextBlob using:

```python
!pip install textblob
```

### NLTK

NLTK is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning.

Install NLTK using:

```python
!pip install nltk
```

Import NLTK and download required packages:

```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')
```

`nltk.download()` opens a GUI to view downloaded packages and update or download new packages manually.

## TextBlob Usage

Import TextBlob and create its object:

```python
from textblob import TextBlob as blob
tb = blob('Hi, please like this post!')
```

### TextBlob Methods

- **tags**: Returns a list of tuples of the form (word, POS tag)
  ```python
  tb.tags
  # [('Hi', 'NNP'), ('please', 'NN'), ('like', 'IN'), ('this', 'DT'), ('post', 'NN')]
  ```

- **noun_phrases**: Returns a list of noun phrases
  ```python
  tb.noun_phrases
  # WordList(['hi'])
  ```

- **sentiment**: Returns a tuple of form (polarity, subjectivity)
  ```python
  tb.sentiment
  # Sentiment(polarity=0.0, subjectivity=0.0)
  ```

Example with positive sentiment:

```python
tb = blob('I love this channel. There are many useful posts here!')
tb.sentiment
# Sentiment(polarity=0.4583333333333333, subjectivity=0.3666666666666667)
```

## Real-Time Voice Recording

Install required packages:

```bash
pip install SpeechRecognition
conda install pyaudio
```

For detailed explanation, refer to the video: [Speech Recognition in Python](https://kgptalkie.com/real-time-sentiment-analysis-of-a-phone-call-using-nltk-and-textblob-in-python)

### Code Implementation

```python
import speech_recognition as sr

r = sr.Recognizer()
with sr.Microphone() as source:
    print('Say Something...')
    audio = r.listen(source, timeout=2)
try:
    text = r.recognize_google(audio)
    tb = blob(text)
    print(text)
    print(tb.sentiment)
except:
    print('Sorry... Try again')
```

Example output:

```
Say Something...
these people are really very poor and I am going to kill everybody I I main they don't deserve to live in this country and then either deserve to live on the planet Earth
Sentiment(polarity=-0.02015151515151517, subjectivity=0.5283333333333333)
```

### Running the Code 10 Times

```python
iter_num = 10
index = 0
while (index < iter_num):
    with sr.Microphone() as source:
        print()
        print('Say Something...')
        audio = r.listen(source, timeout=3)
    try:
        text = r.recognize_google(audio)
        tb = blob(text)
        print(text)
        print(tb.sentiment)
    except:
        print('Sorry... Try again')
    index = index + 1
```

Sample outputs:

```
Say Something...
hello hi baby what's up I am missing you so much
Sentiment(polarity=0.0, subjectivity=0.125)
```

```
Say Something...
do you know I love you so much and have anyone told you that you are the one of the most beautiful girl in the world
Sentiment(polarity=0.5125, subjectivity=0.575)
```

**Source:** https://kgptalkie.com/real-time-sentiment-analysis-of-a-phone-call-using-nltk-and-textblob-in-python

```
Say Something...
ok so have you are you done with your dinner are you going to have your dinner
Sentiment(polarity=0.5, subjectivity=0.5)
```

```
Say Something...
Informatica ok one thing do you know we we have a lot of the things common in between us we like like romantic movies and and you books except Raso these are the really great things between us
Sentiment(polarity=0.25, subjectivity=0.5625)
```

**Source:** https://kgptalkie.com/real-time-sentiment-analysis-of-a-phone-call-using-nltk-and-textblob-in-python

```
Say Something...
yes you are right my apology but do not there to tell me again otherwise I'll kill you also that you ok I also you with the gun in your head and in your heart and don't dare to talk to me like this ever
Sentiment(polarity=0.39285714285714285, subjectivity=0.5178571428571428)
```

```
Say Something...
but still I love you so much I hate you don't talk to me like this and I'll never call you back
Sentiment(polarity=-0.10000000000000002, subjectivity=0.5)
```

**Source:** https://kgptalkie.com/real-time-sentiment-analysis-of-a-phone-call-using-nltk-and-textblob-in-python

```
Say Something...
online now you see here in this sentence when I say that but still I love you so much I hate you and don't talk to me like this and I'll never call you now you can see here there is a negativity in this sentence and it is saying that yes there is any creativity in this sentence super have some 123456 and the 7th 7th time running so what I am talking here its Guna of course printed so let's get it printed and then we'll talk again
Sentiment(polarity=0.01111111111111109, subjectivity=0.7222222222222222)
```

```
Say Something...
Sorry... Try again
```

```
Say Something...
kidding and please do not mind I love you 3000 even I love you goodnight
Sentiment(polarity=0.5, subjectivity=0.6)
```

```
Say Something...
ok now you see here so this is there is polarity at this the positivity Heera 20.5 so this is how you see here our phone call is being converted into word text by using this lesson tutorial you can go and watch speech recognition in Python speed detection in Python at KGP talking you can search and you can get this otherwise I have already given the link for this this listen here you can watch from here as well here
Sentiment(polarity=0.5, subjectivity=0.5)
```

As you can see, for all the statements, the polarity is displayed in real time.