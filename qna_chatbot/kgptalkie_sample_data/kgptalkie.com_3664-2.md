https://kgptalkie.com/3664-2

# NLP: End to End Text Processing for Beginners

Published by [berryedelson](https://kgptalkie.com/author/berryedelson/) on 1 September 2020

## Complete Text Processing for Beginners

Everything we express (either verbally or in written) carries huge amounts of information. The topic we choose, our tone, our selection of words, everything adds some type of information that can be interpreted and value can be extracted from it. In theory, we can understand and even predict human behavior using that information.

But there is a problem: one person may generate hundreds or thousands of words in a declaration, each sentence with its corresponding complexity. If you want to scale and analyze several hundreds, thousands, or millions of people or declarations in a given geography, then the situation is unmanageable.

Data generated from conversations, declarations, or even tweets are examples of unstructured data. Unstructured data doesn’t fit neatly into the traditional row and column structure of relational databases, and represent the vast majority of data available in the actual world. It is messy and hard to manipulate. Nevertheless, thanks to the advances in disciplines like [machine learning](https://kgptalkie.com/3664-2), a big revolution is going on regarding this topic. Nowadays it is no longer about trying to interpret text or speech based on its keywords (the old fashioned mechanical way), but about understanding the meaning behind those words (the cognitive way). This way, it is possible to detect figures of speech like irony or even perform sentiment analysis.

### Natural Language Processing

Natural Language Processing (NLP) is a field of artificial intelligence that gives the machines the ability to read, understand and derive meaning from human languages.

**Ref:** [Watch Full Video](https://kgptalkie.com/3664-2)

**Dataset:** [https://www.kaggle.com/kazanova/sentiment140/data#](https://www.kaggle.com/kazanova/sentiment140/data#)

---

## Installing libraries

**SpaCy** is an open-source software library that is published and distributed under MIT license, and is developed for performing simple to advanced Natural Language Processing (N.L.P) tasks such as:

- tokenization
- part-of-speech tagging
- named entity recognition
- text classification
- calculating semantic similarities between text
- lemmatization
- dependency parsing
- among others

```bash
pip install -U spacy
pip install -U spacy-lookups-data
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
python -m spacy download en_core_web_lg
```

---

## Tasks Overview

In this article, we are going to perform the below tasks:

### General Feature Extraction

- File loading
- Word counts
- Characters count
- Average characters per word
- Stop words count
- Count #HashTags and @Mentions
- If numeric digits are present in twitts
- Upper case word counts

### Preprocessing and Cleaning

- Lower case
- Contraction to Expansion
- Emails removal and counts
- URLs removal and counts
- Removal of **RT**
- Removal of Special Characters
- Removal of multiple spaces
- Removal of HTML tags
- Removal of accented characters
- Removal of Stop Words
- Conversion into base form of words
- Common Occuring words Removal
- Rare Occuring words Removal
- Word Cloud
- Spelling Correction

### Tokenization

- Lemmatization
- Detecting Entities using NER
- Noun Detection
- Language Detection
- Sentence Translation

### Using Inbuilt Sentiment Classifier

### Advanced Text Processing and Feature Extraction

- N-Gram, Bi-Gram etc
- Bag of Words (BoW)
- Term Frequency Calculation
- TF
- Inverse Document Frequency
- IDF
- TFIDF
- Term Frequency – Inverse Document Frequency
- Word Embedding
- Word2Vec using SpaCy

### Machine Learning Models for Text Classification

- SGDClassifier
- LogisticRegression
- LogisticRegressionCV
- LinearSVC
- RandomForestClassifier

---

## Importing libraries

```python
import pandas as pd
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

df = pd.read_csv('twitter16m.csv', encoding='latin1', header=None)
df.head()
```

---

## Data Sample

| 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 0 | 1467810369 | Mon Apr 06 22:19:45 PDT 2009 | NO_QUERY | _TheSpecialOne_ | @switchfoot http://twitpic.com/2y1zl – Awww, t… |
| 1 | 1467810672 | Mon Apr 06 22:19:49 PDT 2009 | NO_QUERY | scotthamilton | is upset that he can’t update his Facebook by … |
| 2 | 1467810917 | Mon Apr 06 22:19:53 PDT 2009 | NO_QUERY | mattycus | @Kenichan I dived many times for the ball. Man… |
| 3 | 1467811184 | Mon Apr 06 22:19:57 PDT 2009 | NO_QUERY | ElleCTF | my whole body feels itchy and like its on fire |
| 4 | 1467811193 | Mon Apr 06 22:19:57 PDT 2009 | NO_QUERY | Karoli | @nationwideclass no, it’s not behaving at all… |

```python
df = df[[5, 0]]
df.columns = ['twitts', 'sentiment']
df.head()
```

| twitts | sentiment |
|--------|-----------|
| @switchfoot http://twitpic.com/2y1zl – Awww, t… | 0 |
| is upset that he can’t update his Facebook by … | 0 |
| @Kenichan I dived many times for the ball. Man… | 0 |
| my whole body feels itchy and like its on fire | 0 |
| @nationwideclass no, it’s not behaving at all… | 0 |

```python
df['sentiment'].value_counts()
```

```
4    800000
0    800000
Name: sentiment, dtype: int64
```

```python
sent_map = {0: 'negative', 4: 'positive'}
```

---

## Word Counts

```python
df['word_counts'] = df['twitts'].apply(lambda x: len(str(x).split()))
df.head()
```

| twitts | sentiment | word_counts |
|--------|-----------|-------------|
| @switchfoot http://twitpic.com/2y1zl – Awww, t… | 0 | 19 |
| is upset that he can’t update his Facebook by … | 0 | 21 |
| @Kenichan I dived many times for the ball. Man… | 0 | 18 |
| my whole body feels itchy and like its on fire | 0 | 10 |
| @nationwideclass no, it’s not behaving at all… | 0 | 21 |

---

## Characters Count

```python
df['char_counts'] = df['twitts'].apply(lambda x: len(x))
df.head()
```

| twitts | sentiment | word_counts | char_counts |
|--------|-----------|-------------|-------------|
| @switchfoot http://twitpic.com/2y键 | 0 | 19 | 115 |
| is upset that he can’t update his Facebook by … | 0 | 21 | 111 |
| @Kenichan I dived many times for the ball. Man… | 0 | 18 | 89 |
| my whole body feels itchy and like its on fire | 0 | 10 | 47 |
| @nationwideclass no, it’s not behaving at all… | 0 | 21 | 111 |

---

## Average Word Length

```python
def get_avg_word_len(x):
    words = x.split()
    word_len = 0
    for word in words:
        word_len = word_len + len(word)
    return word_len / len(words)

df['avg_word_len'] = df['twitts'].apply(lambda x: get_avg_word_len(x))
df.head()
```

| twitts | sentiment | word_counts | char_counts | avg_word_len |
|--------|-----------|-------------|-------------|--------------|
| @switchfoot http://twitpic.com/2y1zl – Awww, t… | 0 | 19 | 115 | 5.052632 |
| is upset that he can’t update his Facebook by … | 0 | 21 | 111 | 4.285714 |
| @Kenichan I dived many times for the ball. Man… | 0 | 18 | 89 | 3.944444 |
| my whole body feels itchy and like its on fire | 0 | 10 | 47 | 3.700000 |
| @nationwideclass no, it’s not behaving at all… | 0 | 21 | 111 | 4.285714 |

---

## Stop Words Count

```python
print(STOP_WORDS)
```

```
{'one', 'up', 'further', 'herself', 'nevertheless', 'their', 'when', 'a', 'bottom', 'both', 'also', 'i', 'sometime', 'ours', "'d", 'him', 'together', 'former', 'hereafter', 'whereby', "'ll", 'three', 'same', 'is', 'say', 'hers', 'must', 'five', 'you', 'across', 'n‘t', 'mostly', 'into', 'am', 'myself', 'something', 'could', 'being', 'seems', 'go', 'only', 'fifteen', 'either', 'us', 'than', 'latter', 'so', 'after', 'name', 'there', 'that', 'next', 'even', 'without', 'along', 'behind', 'very', 'whereas', 'off', 'herein', 'although', 'such', 'themselves', 'then', 'in', 'under', 'of', 'onto', 'really', 'due', 'otherwise', 'give', 'yourself', 'indeed', 'my', 'mine', 'show', 'via', 'elsewhere', 'be', 'just', 'thence', 'them', 'beside', 'though', 'as', 'out', 'third', 'however', 'twelve', 'except', '‘d', 'anything', 'move', 'side', 'everything', 'all', 'towards', 'whatever', 'will', 'n’t', 'toward', 'keep', 'hereupon', 'might', 'no', 'own', 'itself', 'for', 'can', 'rather', 'whether', 'while', 'and', 'part', 'over', 'else', 'has', 'forty', 'about', 'hereby', 'sixty', 'using', 'here', 'please', 'often', '’re', 'any', 'ca', 'per', 'whole', 'it', 'are', 'from', 'had', 'thru', '’m', 'two', 'fifty', 'your', 'latterly', 'again', 'or', 'few', 'against', 'much', 'somewhere', 'but', '’d', 'somehow', 'never', 'becoming', 'down', 'regarding', 'always', 'other', 'amount', 'because', 'noone', 'anyone', 'six', 'each', 'thus', 'alone', 'why', 'his', 'sometimes', 'now', 'since', 'become', 'see', 'she', 'where', 'whereafter', 'various', 'perhaps', 'another', 'who', 'anyhow', 'yourselves', 'someone', 'ten', 'became', 'nothing', 'front', 'an', 'anyway', 'get', 'thereafter', "'re", 'our', 'call', 'therein', 'have', 'this', 'above', 'some', 'namely', '‘re', 'seem', 'until', '’ll', 'more', 'still', "n't", 'the', 'does', 'himself', 'take', 'he', 'which', 'seeming', 'been', 'beforehand', 'may', 'do', 'well', 'ever', 'used', 'enough', 'every', 'top', 'made', "'m", 'hundred', 'almost', 'her', 'moreover', 'wherever', '’s', 'amongst', 'meanwhile', 'nobody', 'ourselves', 'whenever', 'at', 'wherein', 'nowhere', 'around', 'between', 'last', 'others', 'becomes', 'they', 'full', 'below', 'nor', 'before', 'what', 'within', 'these', 'besides', 'whereupon', 'how', 'throughout', 'eight', "'s", 'on', 'most', 'if', '‘ve', 'should', 'four', 'serious', 'thereby', '‘ll', 'whence', 'done', 'anywhere', 'yours', 'formerly', 'everyone', 'whose', 'back', 'make', 'among', 'first', 'we', '‘s', 'neither', 'doing', 'already', 'those', 'empty', 'did', 'not', '‘m', 'less', 'to', 'during', 'twenty', 'too', 'put', 'nine', 'yet', 'everywhere', 'quite', 'were', 'seemed', '’ve', 'through', 'once', 'whither', 'thereupon', 'whoever', "'ve", 'therefore', 'me', 'unless', 'whom', 'cannot', 'afterwards', 'none', 'least', 'hence', 'eleven', 'with', 'upon', 'was', 'would', 'by', 'beyond', 'several', 'its', 'many', 're'}
```

```python
x = 'this is text data'
x.split()
```

```
['this', 'is', 'text', 'data']
```

```python
len([t for t in x.split() if t in STOP_WORDS])
```

```
2
```

```python
df['stop_words_len'] = df['twitts'].apply(lambda x: len([t for t in x.split() if t in STOP_WORDS]))
df.head()
```

| twitts | sentiment | word_counts | char_counts | avg_word_len | stop_words_len |
|--------|-----------|-------------|-------------|--------------|----------------|
| @switchfoot http://twitpic.com/2y1zl – Awww, t… | 0 | 19 | 115 | 5.052632 | 4 |
| is upset that he can’t update his Facebook by … | 0 | 21 | 111 | 4.285714 | 9 |
| @Kenichan I dived many times for the ball. Man… | 0 | 18 | 89 | 3.944444 | 7 |
| my whole body feels itchy and like its on fire | 0 | 10 | 47 | 3.700000 | 5 |
| @nationwideclass no, it’s not behaving at all… | 0 | 21 | 111 | 4.285714 | 10 |

---

## Count #HashTags and @Mentions

```python
x = 'this #hashtag and this is @mention'
[t for t in x.split() if t.startswith('@')]
```

```
['@mention']
```

```python
df['hashtags_count'] = df['twitts'].apply(lambda x: len([t for t in x.split() if t.startswith('#')]))
df['mentions_count'] = df['twitts'].apply(lambda x: len([t for t in x.split() if t.startswith('@')]))
df.head()
```

| twitts | sentiment | word_counts | char_counts | avg_word_len | stop_words_len | hashtags_count | mentions_count |
|--------|-----------|-------------|-------------|--------------|----------------|----------------|----------------|
| @switchfoot http://twitpic.com/2y1zl – Awww, t… | 0 | 19 | 115 | 5.052632 | 4 | 0 | 1 |
| is upset that he can’t update his Facebook by … | 0 | 21 | 111 | 4.285714 | 9 | 0 | 0 |
| @Kenichan I dived many times for the ball. Man… | 0 | 18 | 89 | 3.944444 | 7 | 0 | 1 |
| my whole body feels itchy and like its on fire | 0 | 10 | 47 | 3.700000 | 5 | 0 | 0 |
| @nationwideclass no, it’s not behaving at all… | 0 | 21 | 111 | 4.285714 | 10 | 0 | 1 |

---

## If numeric digits are present in twitts

```python
df['numerics_count'] = df['twitts'].apply(lambda x: len([t for t in x.split() if t.isdigit()]))
df.head()
```

| twitts | sentiment | word_counts | char_counts | avg_word_len | stop_words_len | hashtags_count | mentions_count | numerics_count |
|--------|-----------|-------------|-------------|--------------|----------------|----------------|----------------|----------------|
| @switchfoot http://twitpic.com/2y1zl – Awww, t… | 0 | 19 | 115 | 5.052632 | 4 | 0 | 1 | 0 |
| is upset that he can’t update his Facebook by … | 0 | 21 | 111 | 4.285714 | 9 | 0 | 0 | 0 |
| @Kenichan I dived many times for the ball. Man… | 0 | 18 | 89 | 3.944444 | 7 | 0 | 1 | 0 |
| my whole body feels itchy and like its on fire | 0 | 10 | 47 | 3.700000 | 5 | 0 | 0 | 0 |
| @nationwideclass no, it’s not behaving at all… | 0 | 21 | 111 | 4.285714 | 10 | 0 | 1 | 0 |

---

## UPPER case words count

```python
df['upper_counts'] = df['twitts'].apply(lambda x: len([t for t in x.split() if t.isupper() and len(x) > 3]))
df.head()
```

| twitts | sentiment | word_counts | char_counts | avg_word_len | stop_words_len | hashtags_count | mentions_count | numerics_count | upper_counts |
|--------|-----------|-------------|-------------|--------------|----------------|----------------|----------------|----------------|--------------|
| @switchfoot http://twitpic.com/2y1zl – Awww, t… | 0 | 19 | 115 | 5.052632 | 4 | 0 | 1 | 0 | 1 |
| is upset that he can’t update his Facebook by … | 0 | 21 | 111 | 4.285714 | 9 | 0 | 0 | 0 | 0 |
| @Kenichan I dived many times for the ball. Man… | 0 | 18 | 89 | 3.944444 | 7 | 0 | 1 | 0 | 1 |
| my whole body feels itchy and like its on fire | 0 | 10 | 47 | 3.700000 | 5 | 0 | 0 | 0 | 0 |
| @nationwideclass no, it’s not behaving at all… | 0 | 21 | 111 | 4.285714 | 10 | 0 | 1 | 0 | 1 |

---

## Preprocessing and Cleaning

### Lower case conversion

```python
df['twitts'] = df['twitts'].apply(lambda x: x.lower())
df.head(2)
```

| twitts | sentiment | word_counts | char_counts | avg_word_len | stop_words_len | hashtags_count | mentions_count | numerics_count | upper_counts |
|--------|-----------|-------------|-------------|--------------|----------------|----------------|----------------|----------------|--------------|
| @switchfoot http://twitpic.com/2y1zl – awww, t… | 0 | 19 | 115 | 5.052632 | 4 | 0 | 1 | 0 | 1 |
| is upset that he cannot update his facebook by … | 0 | 21 | 111 | 4.285714 | 9 | 0 | 0 | 0 | 0 |

---

## Contraction to Expansion

```python
contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how does",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    " u ": " you ",
    " ur ": " your ",
    " n ": " and "
}

def cont_to_exp(x):
    if type(x) is str:
        for key in contractions:
            value = contractions[key]
            x = x.replace(key, value)
        return x
    else:
        return x

x = "hi, i'd be happy"
cont_to_exp(x)
```

```
'hi, i would be happy'
```

```python
%%time
df['twitts'] = df['twitts'].apply(lambda x: cont_to_exp(x))
```

```
Wall time: 52.7 s
```

```python
df.head()
```

| twitts | sentiment | word_counts | char_counts | avg_word_len | stop_words_len | hashtags_count | mentions_count | numerics_count | upper_counts |
|--------|-----------|-------------|-------------|--------------|----------------|----------------|----------------|----------------|--------------|
| @switchfoot http://twitpic.com/2y1zl – awww, t… | 0 | 19 | 115 | 5.052632 | 4 | 0 | 1 | 0 | 1 |
| is upset that he cannot update his facebook by… | 0 | 21 | 111 | 4.285714 | 9 | 0 | 0 | 0 | 0 |
| @kenichan i dived many times for the ball. man… | 0 | 18 | 89 | 3.944444 | 7 | 0 | 1 | 0 | 1 |
| my whole body feels itchy and like its on fire | 0 | 10 | 47 | 3.700000 | 5 | 0 | 0 | 0 | 0 |
| @nationwideclass no, it is not behaving at all… | 0 | 21 | 111 | 4.285714 | 10 | 0 | 1 | 0 | 1 |

---

## Count and Remove Emails

```python
import re
x = 'hi my email me at email@email.com another@email.com'
re.findall(r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)', x)
```

```
['email@email.com', 'another@email.com']
```

```python
df['emails'] = df['twitts'].apply(lambda x: re.findall(r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)', x))
df['emails_count'] = df['emails'].apply(lambda x: len(x))
df[df['emails_count'] > 0].head()
```

| twitts | sentiment | word_counts | char_counts | avg_word_len | stop_words_len | hashtags_count | mentions_count | numerics_count | upper_counts | emails | emails_count |
|--------|-----------|-------------|-------------|--------------|----------------|----------------|----------------|----------------|--------------|--------|--------------|
| i want a new laptop. hp tx2000 is the bomb. :… | 0 | 20 | 103 | 4.150000 | 6 | 0 | 0 | 0 | 4 | [gabbehhramos@yahoo.com] | 1 |
| who stole elledell@gmail.com? | 0 | 3 | 31 | 9.000000 | 1 | 0 | 0 | 0 | 0 | [elledell@gmail.com] | 1 |
| @alexistehpom really? did you send out all th… | 0 | 20 | 130 | 5.500000 | 11 | 0 | 1 | 0 | 0 | [missataari@gmail.com] | 1 |
| @laureystack awh…that is kinda sad lol add … | 0 | 8 | 76 | 8.500000 | 0 | 0 | 1 | 0 | 0 | [hello.kitty.65@hotmail.com] | 1 |
| @jilliancyork got 2 bottom of it, human error… | 0 | 21 | 137 | 5.428571 | 7 | 0 | 1 | 1 | 0 | [press@linkedin.com] | 1 |

```python
re.sub(r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)', '', x)
```

```
'hi my email me at  '
```

```python
df['twitts'] = df['twitts'].apply(lambda x: re.sub(r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)', '', x))
df[df['emails_count'] > 0].head()
```

| twitts | sentiment | word_counts | char_counts | avg_word_len | stop_words_len | hashtags_count | mentions_count | numerics_count | upper_counts | emails | emails_count |
|--------|-----------|-------------|-------------|--------------|----------------|----------------|----------------|----------------|--------------|--------|--------------|
| i want a new laptop. hp tx2000 is the bomb. :… | 0 | 20 | 103 | 4.150000 | 6 | 0 | 0 | 0 | 4 | [gabbehhramos@yahoo.com] | 1 |
| who stole ? | 0 | 3 | 31 | 9.000000 | 1 | 0 | 0 | 0 | 0 | [elledell@gmail.com] | 1 |
| @alexistehpom really? did you send out all th… | 0 | 20 | 130 | 5.500000 | 11 | 0 | 1 | 0 | 0 | [missataari@gmail.com] | 1 |
| @laureystack awh…that is kinda sad lol add … | 0 | 8 | 76 | 8.500000 | 0 | 0 | 1 | 0 | 0 | [hello.kitty.65@hotmail.com] | 1 |
| @jilliancyork got 2 bottom of it, human error… | 0 | 21 | 137 | 5.428571 | 7 | 0 | 1 | 1 | 0 | [press@linkedin.com] | 1 |

---

## Count URLs and Remove it

```python
x = 'hi, to watch more visit https://youtube.com/kgptalkie'
re.findall(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', x)
```

```
[('https', 'youtube.com', '/kgptalkie')]
```

```python
df['urls_flag'] = df['twitts'].apply(lambda x: len(re.findall(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', x)))
re.sub(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', x)
```

```
'hi, to watch more visit '
```

```python
df['twitts'] = df['twitts'].apply(lambda x: re.sub(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', x))
df.head()
```

| twitts | sentiment | word_counts | char_counts | avg_word_len | stop_words_len | hashtags_count | mentions_count | numerics_count | upper_counts | emails | emails_count | urls_flag |
|--------|-----------|-------------|-------------|--------------|----------------|----------------|----------------|----------------|--------------|--------|--------------|-----------|
| @switchfoot – awww that is a bummer. you sh… | 0 | 19 | 115 | 5.052632 | 4 | 0 | 1 | 0 | 1 | [] | 0 | 1 |
| is upset that he cannot update his facebook by… | 0 | 21 | 111 | 4.285714 | 9 | 0 | 0 | 0 | 0 | [] | 0 | 0 |
| @kenichan i dived many times for the ball. man… | 0 | 18 | 89 | 3.944444 | 7 | 0 | 1 | 0 | 1 | [] | 0 | 0 |
| my whole body feels itchy and like its on fire | 0 | 10 | 47 | 3.700000 | 5 | 0 | 0 | 0 | 0 | [] | 0 | 0 |
| @nationwideclass no, it is not behaving at all… | 0 | 21 | 111 | 4.285714 | 10 | 0 | 1 | 0 | 1 | [] | 0 | 0 |

---

## Remove RT

```python
df['twitts'] = df['twitts'].apply(lambda x: re.sub('RT', "", x))
```

---

## Special Chars removal or punctuation removal

```python
df['twitts'] = df['twitts'].apply(lambda x: re.sub('[^A-Z a-z 0-9-]+', '', x))
df.head()
```

| twitts | sentiment | word_counts | char_counts | avg_word_len | stop_words_len | hashtags_count | mentions_count | numerics_count | upper_counts | emails | emails_count | urls_flag |
|--------|-----------|-------------|-------------|--------------|----------------|----------------|----------------|----------------|--------------|--------|--------------|-----------|
| switchfoot – awww that is a bummer you shoul… | 0 | 19 | 115 | 5.052632 | 4 | 0 | 1 | 0 | 1 | [] | 0 | 1 |
| is upset that he cannot update his facebook by… | 0 | 21 | 111 | 4.285714 | 9 | 0 | 0 | 0 | 0 | [] | 0 | 0 |
| kenichan i dived many times for the ball manag… | 0 | 18 | 89 | 3.944444 | 7 | 0 | 1 | 0 | 1 | [] | 0 | 0 |
| my whole body feels itchy and like its on fire | 0 | 10 | 47 | 3.700000 | 5 | 0 | 0 | 0 | 0 | [] | 0 | 0 |
| nationwideclass no it is not behaving at all i… | 0 | 21 | 111 | 4.285714 | 10 | 0 | 1 | 0 | 1 | [] | 0 | 0 |

---

## Remove multiple spaces

```python
x = 'thanks    for    watching and    please    like this video'
" ".join(x.split())
```

```
'thanks for watching and please like this video'
```

```python
df['twitts'] = df['twitts'].apply(lambda x: " ".join(x.split()))
df.head(2)
```

| twitts | sentiment | word_counts | char_counts | avg_word_len | stop_words_len | hashtags_count | mentions_count | numerics_count | upper_counts | emails | emails_count | urls_flag |
|--------|-----------|-------------|-------------|--------------|----------------|----------------|----------------|----------------|--------------|--------|--------------|-----------|
| switchfoot – awww that is a bummer you shoulda… | 0 | 19 | 115 | 5.052632 | 4 | 0 | 1 | 0 | 1 | [] | 0 | 1 |
| is upset that he cannot update his facebook by… | 0 | 21 | 111 | 4.285714 | 9 | 0 | 0 | 0 | 0 | [] | 0 | 0 |

---

## Remove HTML tags

```python
from bs4 import BeautifulSoup
x = '<html><h2>Thanks for watching</h2></html>'
BeautifulSoup(x, 'lxml').get_text()
```

```
'Thanks for watching'
```

```python
%%time
df['twitts'] = df['twitts'].apply(lambda x: BeautifulSoup(x, 'lxml').get_text())
```

```
Wall time: 11min 37s
```

---

## Remove Accented Chars

```python
import unicodedata
x = 'Áccěntěd těxt'
def remove_accented_chars(x):
    x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return x

remove_accented_chars(x)
```

```
'Accented text'
```

---

## SpaCy and NLP

### Remove Stop Words

```python
import spacy
x = 'this is stop words removal code is a the an how what'
" ".join([t for t in x.split() if t not in STOP_WORDS])
```

```
'stop words removal code'
```

```python
df['twitts'] = df['twitts'].apply(lambda x: " ".join([t for t in x.split() if t not in STOP_WORDS]))
df.head()
```

| twitts | sentiment | word_counts | char_counts | avg_word_len | stop_words_len | hashtags_count | mentions_count | numerics_count | upper_counts | emails | emails_count | urls_flag |
|--------|-----------|-------------|-------------|--------------|----------------|----------------|----------------|----------------|--------------|--------|--------------|-----------|
| switchfoot – awww bummer shoulda got david car… | 0 | 19 | 115 | 5.052632 | 4 | 0 | 1 | 0 | 1 | [] | 0 | 1 |
| upset update facebook texting cry result schoo… | 0 | 21 | 111 | 4.285714 | 9 | 0 | 0 | 0 | 0 | [] | 0 | 0 |
| kenichan dived times ball managed save 50 rest… | 0 | 18 | 89 | 3.944444 | 7 | 0 | 1 | 0 | 1 | [] | 0 | 0 |
| body feels itchy like fire | 0 | 10 | 47 | 3.700000 | 5 | 0 | 0 | 0 | 0 | [] | 0 | 0 |
| nationwideclass behaving mad | 0 | 21 | 111 | 4.285714 | 10 | 0 | 1 | 0 | 1 | [] | 0 | 0 |

---

## Convert into base or root form of word

```python
nlp = spacy.load('en_core_web_sm')
x = 'kenichan dived times ball managed save 50 rest'
def make_to_base(x):
    x_list = []
    doc = nlp(x)
    for token in doc:
        lemma = str(token.lemma_)
        if lemma == '-PRON-' or lemma == 'be':
            lemma = token.text
        x_list.append(lemma)
    print(" ".join(x_list))

make_to_base(x)
```

```
kenichan dive time ball manage save 50 rest
```

---

## Common words removal

```python
' '.join(df.head()['twitts'])
```

```
'switchfoot - awww bummer shoulda got david carr day d upset update facebook texting cry result school today blah kenichan dived times ball managed save 50 rest bounds body feels itchy like fire nationwideclass behaving mad'
```

```python
text = ' '.join(df['twitts'])
text = text.split()
freq_comm = pd.Series(text).value_counts()
f20 = freq_comm[:20]
f20
```

```
good      89366
day       82299
like      77735
-         69662
today     64512
going     64078
love      63421
work      62804
got       60749
time      56081
lol       55094
know      51172
im        50147
want      42070
new       41995
think     41040
night     41029
amp       40616
thanks    39311
home      39168
dtype: int64
```

```python
df['twitts'] = df['twitts'].apply(lambda x: " ".join([t for t in x.split() if t not in f20]))
```

---

## Rare words removal

```python
rare20 = freq_comm[-20:]
rare20
```

```
veru              1
80-90f            1
refrigerant       1
demaisss          1
knittingsci-fi    1
wendireed         1
danielletuazon    1
chacha8           1
a-zquot           1
krustythecat      1
westmount         1
-appreciate       1
motocycle         1
madamhow          1
felspoon          1
fastbloke         1
900pmno           1
nxec              1
laassssttt        1
update-uri        1
dtype: int64
```

```python
rare = freq_comm[freq_comm.values == 1]
rare
```

```
mamat             1
fiive             1
music-festival    1
leenahyena        1
11517             1
                 ..
fastbloke         1
900pmno           1
nxec              1
laassssttt        1
update-uri        1
Length: 536196, dtype: int64
```

```python
df['twitts'] = df['twitts'].apply(lambda x: ' '.join([t for t in x.split() if t not in rare20]))
df.head()
```

| twitts | sentiment | word_counts | char_counts | avg_word_len | stop_words_len | hashtags_count | mentions_count | numerics_count | upper_counts | emails | emails_count | urls_flag |
|--------|-----------|-------------|-------------|--------------|----------------|----------------|----------------|----------------|--------------|--------|--------------|-----------|
| switchfoot awww bummer shoulda david carr d | 0 | 19 | 115 | 5.052632 | 4 | 0 | 1 | 0 | 1 | [] | 0 | 1 |
| upset update facebook texting cry result schoo… | 0 | 21 | 111 | 4.285714 | 9 | 0 | 0 | 0 | 0 | [] | 0 | 0 |
| kenichan dived times ball managed save 50 rest… | 0 | 18 | 89 | 3.944444 | 7 | 0 | 1 | 0 | 1 | [] | 0 | 0 |
| body feels itchy fire | 0 | 10 | 47 | 3.700000 | 5 | 0 | 0 | 0 | 0 | [] | 0 | 0 |
| nationwideclass behaving mad | 0 | 21 | 111 | 4.285714 | 10 | 0 | 1 | 0 | 1 | [] | 0 | 0 |

---

## Word Cloud Visualization

```python
# !pip install wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
%matplotlib inline
x = ' '.join(text[:20000])
len(text)
```

```
10837079
```

```python
wc = WordCloud(width=800, height=400).generate(x)
plt.imshow(wc)
plt.axis('off')
plt.show()
```

---

## Spelling Correction

```python
# !pip install -U textblob
# !python -m textblob.download_corpora
from textblob import TextBlob
x = 'tanks forr waching this vidio carri'
x = TextBlob(x).correct()
x
```

```
TextBlob("tanks for watching this video carry")
```

---

## Tokenization

```python
x = 'thanks#watching this video. please like it'
TextBlob(x).words
```

```
WordList(['thanks', 'watching', 'this', 'video', 'please', 'like', 'it'])
```

```python
doc = nlp(x)
for token in doc:
    print(token)
```

```
thanks#watching
this
video
.
please
like
it
```

---

## Lemmatization

```python
x = 'runs run running ran'
from textblob import Word
for token in x.split():
    print(Word(token).lemmatize())
```

```
run
run
running
ran
```

```python
doc = nlp(x)
for token in doc:
    print(token.lemma_)
```

```
run
run
run
run
```

---

## Detect Entities using NER of SpaCy

```python
x = "Breaking News: Donald Trump, the president of the USA is looking to sign a deal to mine the moon"
doc = nlp(x)
for ent in doc.ents:
    print(ent.text + ' - ' + ent.label_ + ' - ' + str(spacy.explain(ent.label_)))
```

```
Donald Trump - PERSON - People, including fictional
USA - GPE - Countries, cities, states
```

```python
from spacy import displacy
displacy.render(doc, style='ent')
```

```
Breaking News:
Donald Trump
PERSON
, the president of the
USA
GPE
is looking to sign a deal to mine the moon
```

---

## Detecting Nouns

```python
x = 'Breaking News: Donald Trump, the president of the USA is looking to sign a deal to mine the moon'
for noun in doc.noun_chunks:
    print(noun)
```

```
Breaking News
Donald Trump
the president
the USA
a deal
the moon
```

---

## Translation and Language Detection

```python
x = 'Breaking News: Donald Trump, the president of the USA is looking to sign a deal to mine the moon'
tb = TextBlob(x)
tb.detect_language()
```

```
'en'
```

```python
tb.translate(to='bn')
```

```
TextBlob("ব্রেকিং নিউজ: যুক্তরাষ্ট্রের রাষ্ট্রপতি ডোনাল্ড ট্রাম্প চাঁদটি খনির জন্য একটি চুক্তিতে সই করতে চাইছেন")
```

---

## Use inbuilt sentiment classifier

```python
from textblob.sentiments import NaiveBayesAnalyzer
x = 'we all stands together to fight with corona virus. we will win together'
tb = TextBlob(x, analyzer=NaiveBayesAnalyzer())
tb.sentiment
```

```
Sentiment(classification='pos', p_pos=0.8259779151942094, p_neg=0.17402208480578962)
```

```python
x = 'we all are sufering from corona'
tb = TextBlob(x, analyzer=NaiveBayesAnalyzer())
tb.sentiment
```

```
Sentiment(classification='pos', p_pos=0.75616044472398, p_neg=0.2438395552760203)
```

---

## Advanced Text Processing

### N-Grams

```python
x = 'thanks for watching'
tb = TextBlob(x)
tb.ngrams(3)
```

```
[WordList(['thanks', 'for', 'watching'])]
```

---

### Bag of Words

```python
x = ['this is first sentence this is', 'this is second', 'this is last']
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(1,1))
text_counts = cv.fit_transform(x)
text_counts.toarray()
```

```
array([[1, 2, 0, 0, 1, 2],
       [0, 1, 0, 1, 0, 1],
       [0, 1, 1, 0, 0, 1]], dtype=int64)
```

```python
cv.get_feature_names()
```

```
['first', 'is', 'last', 'second', 'sentence', 'this']
```

```python
bow = pd.DataFrame(text_counts.toarray(), columns=cv.get_feature_names())
bow
```

| first | is | last | second | sentence | this |
|-------|----|------|--------|----------|------|
| 1     | 2  | 0    | 0      | 1        | 2    |
| 0     | 1  | 0    | 1      | 0        | 1    |
| 0     | 1  | 1    | 0      | 0        | 1    |

---

## Term Frequency

```python
tf = bow.copy()
for index, row in enumerate(tf.iterrows()):
    for col in row[1].index:
        tf.loc[index, col] = tf.loc[index, col]/sum(row[1].values)
tf
```

| first | is | last | second | sentence | this |
|-------|----|------|--------|----------|------|
| 0.166667 | 0.333333 | 0.000000 | 0.000000 | 0.166667 | 0.333333 |
| 0.000000 | 0.333333 | 0.000000 | 0.333333 | 0.000000 | 0.333333 |
| 0.000000 | 0.333333 | 0.333333 | 0.000000 | 0.000000 | 0.333333 |

---

## Inverse Document Frequency IDF

```python
import numpy as np
x_df = pd.DataFrame(x, columns=['words'])
x_df
```

| words |
|-------|
| this is first sentence this is |
| this is second |
| this is last |

```python
N = bow.shape[0]
N
```

```
3
```

```python
bb = bow.astype('bool')
bb
```

| first | is | last | second | sentence | this |
|-------|----|------|--------|----------|------|
| True  | True | False | False | True | True |
| False | True | False | True | False | True |
| False | True | True | False | False | True |

```python
bb['is'].sum()
```

```
3
```

```python
cols = bb.columns
cols
```

```
Index(['first', 'is', 'last', 'second', 'sentence', 'this'], dtype='object')
```

```python
nz = []
for col in cols:
    nz.append(bb[col].sum())
nz
```

```
[1, 3, 1, 1, 1, 3]
```

```python
idf = []
for index, col in enumerate(cols):
    idf.append(np.log((N + 1)/(nz[index] + 1)) + 1)
idf
```

```
[1.6931471805599454, 1.0, 1.6931471805599454, 1.6931471805599454, 1.6931471805599454, 1.0]
```

---

## TFIDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
x_tfidf = tfidf.fit_transform(x_df['words'])
x_tfidf.toarray()
```

```
array([[0.45688214, 0.5396839 , 0.        , 0.        , 0.45688214,
        0.5396839 ],
       [0.        , 0.45329466, 0.        , 0.76749457, 0.        ,
        0.45329466],
       [0.        , 0.45329466, 0.76749457, 0.        , 0.        ,
        0.45329466]])
```

```python
tfidf.idf_
```

```
array([1.69314718, 1.        , 1.69314718, 1.69314718, 1.69314718,
       1.        ])
```

---

## Word Embeddings

### Word2Vec

```python
# !python -m spacy download en_core_web_lg
nlp = spacy.load('en_core_web_lg')
doc = nlp('thank you! dog cat lion dfasaa')
for token in doc:
    print(token.text, token.has_vector)
```

```
thank True
you True
! True
dog True
cat True
lion True
dfasaa False
```

```python
token.vector.shape
```

```
(300,)
```

```python
nlp('cat').vector.shape
```

```
(300,)
```

```python
for token1 in doc:
    for token2 in doc:
        print(token1.text, token2.text, token1.similarity(token2))
    print()
```

```
thank thank 1.0
thank you 0.5647585
thank ! 0.52147406
thank dog 0.2504265
thank cat 0.20648485
thank lion 0.13629764
C:\ProgramData\Anaconda3\lib\runpy.py:193: UserWarning: [W008] Evaluating Token.similarity based on empty vectors.
  "__main__", mod_spec)
thank dfasaa 0.0

you thank 0.5647585
you you 1.0
you ! 0.4390223
you dog 0.36494097
you cat 0.3080798
you lion 0.20392051
C:\ProgramData\Anaconda3\lib\runpy.py:193: UserWarning: [W008] Evaluating Token.similarity based on empty vectors.
  "__main__", mod_spec)
you dfasaa 0.0

! thank 0.52147406
! you 0.4390223
! ! 1.0
! dog 0.29852203
! cat 0.29702348
! lion 0.19601382
C:\ProgramData\Anaconda3\lib\runpy.py:193: UserWarning: [W008] Evaluating Token.similarity based on empty vectors.
  "__main__", mod_spec)
! dfasaa 0.0

dog thank 0.2504265
dog you 0.36494097
dog ! 0.29852203
dog dog 1.0
dog cat 0.80168545
dog lion 0.47424486
C:\ProgramData\Anaconda3\lib\runpy.py:193: UserWarning: [W008] Evaluating Token.similarity based on empty vectors.
  "__main__", mod_spec)
dog dfasaa 0.0

cat thank 0.20648485
cat you 0.3080798
cat ! 0.29702348
cat dog 0.80168545
cat cat 1.0
cat lion 0.52654374
C:\ProgramData\Anaconda3\lib\runpy.py:193: UserWarning: [W008] Evaluating Token.similarity based on empty vectors.
  "__main__", mod_spec)
cat dfasaa 0.0

lion thank 0.13629764
lion you 0.20392051
lion ! 0.19601382
lion dog 0.47424486
lion cat 0.52654374
lion lion 1.0
C:\ProgramData\Anaconda3\lib\runpy.py:193: UserWarning: [W008] Evaluating Token.similarity based on empty vectors.
  "__main__", mod_spec)
lion dfasaa 0.0
C:\ProgramData\Anaconda3\lib\runpy.py:193: UserWarning: [W008] Evaluating Token.similarity based on empty vectors.
  "__main__", mod_spec)
dfasaa thank 0.0
C:\ProgramData\Anaconda3\lib\runpy.py:193: UserWarning: [W008] Evaluating Token.similarity based on empty vectors.
  "__main__", mod_spec)
dfasaa you 0.0
C:\ProgramData\Anaconda3\lib\runpy.py:193: UserWarning: [W008] Evaluating Token.similarity based on empty vectors.
  "__main__", mod_spec)
dfasaa ! 0.0
C:\ProgramData\Anaconda3\lib\runpy.py:193: UserWarning: [W008] Evaluating Token.similarity based on empty vectors.
  "__main__", mod_spec)
dfasaa dog 0.0
C:\ProgramData\Anaconda3\lib\runpy.py:193: UserWarning: [W008] Evaluating Token.similarity based on empty vectors.
  "__main__", mod_spec)
dfasaa cat 0.0
C:\ProgramData\Anaconda3\lib\runpy.py:193: UserWarning: [W008] Evaluating Token.similarity based on empty vectors.
  "__main__", mod_spec)
dfasaa lion 0.0
dfasaa dfasaa 1.0
```

---

## Machine Learning Models for Text Classification

### BoW

```python
df.shape
```

```
(1600000, 13)
```

```python
df0 = df[df['sentiment'] == 0].sample(2000)
df4 = df[df['sentiment'] == 4].sample(2000)
dfr = df0.append(df4)
dfr.shape
```

```
(4000, 13)
```

```python
dfr_feat = dfr.drop(labels=['twitts','sentiment','emails'], axis=1).reset_index(drop=True)
dfr_feat
```

| word_counts | char_counts | avg_word_len | stop_words_len | hashtags_count | mentions_count | numerics_count | upper_counts | emails_count | urls_flag |
|-------------|-------------|--------------|----------------|----------------|----------------|----------------|--------------|--------------|-----------|
| 15          | 81          | 4.400000     | 6              | 0              | 0              | 0              | 0            | 0          | 0         |
| 8           | 47          | 4.875000     | 4              | 0              | 1              | 0              | 0            | 0          | 0         |
| 15          | 69          | 3.600000     | 6              | 0              | 1              | 0              | 0            | 0          | 0         |
| 9           | 42          | 3.666667     | 4              | 0              | 0              | 0              | 2            | 0          | 0         |
| 14          | 77          | 4.500000     | 5              | 0              | 0              | 0              | 0            | 0          | 0         |

```python
y = dfr['sentiment']
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
text_counts = cv.fit_transform(dfr['twitts'])
text_counts.toarray().shape
```

```
(4000, 9750)
```

```python
dfr_bow = pd.DataFrame(text_counts.toarray(), columns=cv.get_feature_names())
dfr_bow.head(2)
```

| 007peter | 05 | 060594 | 09 | 10 | 100 | 1000 | 10000000000000000000000000000 | 1038 | 1041 | … | zomg | zonked | zoo | zooey | zrovna | zshare | zsk | zwel | zzz | zzzzz |
|----------|----|--------|----|----|-----|------|-----------------------------|------|------|---|------|--------|-----|-------|--------|--------|----|-----|-----|-------|
| 0        | 0  | 0      | 0  | 0  | 0   | 0    | 0                           | 0    | 0    | … | 0    | 0      | 0   | 0     | 0      | 0      | 0  | 0   | 0   | 0     |
| 0        | 0  | 0      | 0  | 0  | 0   | 0    | 0                           | 0    | 0    | … | 0    | 0      | 0   | 0     | 0      | 0      | 0  | 0   | 0   | 0     |

---

## ML Algorithms

```python
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler

sgd = SGDClassifier(n_jobs=-1, random_state=42, max_iter=200)
lgr = LogisticRegression(random_state=42, max_iter=200)
lgrcv = LogisticRegressionCV(cv=2, random_state=42, max_iter=1000)
svm = LinearSVC(random_state=42, max_iter=200)
rfc = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=200)
clf = {'SGD': sgd, 'LGR': lgr, 'LGR-CV': lgrcv, 'SVM': svm, 'RFC': rfc}
clf.keys()
```

```
dict_keys(['SGD', 'LGR', 'LGR-CV', 'SVM', 'RFC'])
```

```python
def classify(X, y):
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    for key in clf.keys():
        clf[key].fit(X_train, y_train)
        y_pred = clf[key].predict(X_test)
        ac = accuracy_score(y_test, y_pred)
        print(key, " ---> ", ac)

%%time
classify(dfr_bow, y)
```

```
SGD  --->  0.62375
LGR  --->  0.65375
LGR-CV  --->  0.6525
SVM  --->  0.6325
RFC  --->  0.6525
Wall time: 1min 42s
```

---

## Manual Feature

```python
dfr_feat.head(2)
```

| word_counts | char_counts | avg_word_len | stop_words_len | hashtags_count | mentions_count | numerics_count | upper_counts | emails_count | urls_flag |
|-------------|-------------|--------------|----------------|----------------|----------------|----------------|--------------|--------------|-----------|
| 15          | 81          | 4.400000     | 6              | 0              | 0              | 0              | 0            | 0          | 0         |
| 8           | 47          | 4.875000     | 4              | 0              | 1              | 0              | 0            | 0          | 0         |

```python
%%time
classify(dfr_feat, y)
```

```
SGD  --->  0.64125
LGR  --->  0.645
LGR-CV  --->  0.65
SVM  --->  0.6475
RFC  --->  0.5675
Wall time: 1.35 s
```

---

## Manual + Bow

```python
X = dfr_feat.join(dfr_bow)
%%time
classify(X, y)
```

```
SGD  --->  0.64875
LGR  --->  0.67125
LGR-CV  --->  0.66125
SVM  --->  0.64375
RFC  --->  0.705
Wall time: 1min 18s
```

---

## TFIDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer
dfr.shape
```

```
(4000, 13)
```

```python
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(dfr['twitts'])
%%time
classify(pd.DataFrame(X.toarray()), y)
```

```
SGD  --->  0.635
LGR  --->  0.65125
LGR-CV  --->  0.6475
SVM  --->  0.63875
RFC  --->  0.6425
Wall time: 1min 37s
```

---

## Word2Vec

```python
def get_vec(x):
    doc = nlp(x)
    return doc.vector.reshape(1, -1)

%%time
dfr['vec'] = dfr['twitts'].apply(lambda x: get_vec(x))
```

```
Wall time: 51.8 s
```

```python
X = np.concatenate(dfr['vec'].to_numpy(), axis=0)
X.shape
```

```
(4000, 300)
```

```python
classify(pd.DataFrame(X), y)
```

```
SGD  --->  0.5925
LGR  --->  0.70625
LGR-CV  --->  0.69375
C:\Users\Laxmi\AppData\Roaming\Python\Python37\site-packages\sklearn\svm\_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
SVM  --->  0.70125
RFC  --->  0.66625
```

```python
def predict_w2v(x):
    for key in clf.keys():
        y_pred = clf[key].predict(get_vec(x))
        print(key, "-->", y_pred)

predict_w2v('hi, thanks for watching this video. please like and subscribe')
```

```
SGD --> [0]
LGR --> [4]
LGR-CV --> [0]
SVM --> [4]
RFC --> [0]
```

```python
predict_w2v('please let me know if you want more video')
```

```
SGD --> [0]
LGR --> [0]
LGR-CV --> [0]
SVM --> [0]
RFC --> [0]
```

```python
predict_w2v('congratulation looking good congrats')
```

```
SGD --> [4]
LGR --> [4]
LGR-CV --> [4]
SVM --> [4]
RFC --> [0]
```

---

## Summary

1. In this article, firstly we have cleared the texts like **removing URLs** and various **tags**.
2. Also, we have used various text featurization techniques like **bag-of-words**, **tf-idf** and **word2vec**.
3. After doing text featurization, we building machine learning models on top of those features.

**Source:** https://kgptalkie.com/3664-2