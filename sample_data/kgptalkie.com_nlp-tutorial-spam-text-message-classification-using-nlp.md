# [NLP Tutorial – Spam Text Message Classification using NLP](https://kgptalkie.com/nlp-tutorial-spam-text-message-classification-using-nlp)

**Published by**  
ignacioberrios  
**on**  
24 August 2020  

---

## Objective

Our objective of this code is to classify texts into two classes: **spam** and **ham**.

---

## What is Natural Language Processing

**Natural Language Processing (NLP)** is the field of Artificial Intelligence, where we analyse text using **machine learning** models.

---

## Application of NLP

- Text Classification
- Spam Filters
- Voice text messaging
- Sentiment analysis
- Spell or grammar check
- Chat bot
- Search Suggestion
- Search Autocorrect
- Automatic Review Analysis system
- Machine translation
- And so much more

---

## Natural Language Understanding (Text classification)

### The Process of Natural Language Understanding (Text Classification)

- Sentence Breakdown
- Natural Language Generation

---

## How to get started with NLP

Following are the libraries which are generally used in Natural Language Processing.

- **Sklearn**
- **Spacy**
- **NLTK**
- **Gensim**
- **Tensorflow and Keras**

```python
!pip install scikit-learn
!python -m spacy download en
!pip install -U spacy
!pip install gensim
!pip install lightgbm
```

---

## Application of these libraries

- Tokenization
- Parts of Speech Tagging
- Entity Detection
- Dependency Parsing
- Noun Phrases
- Words-to-Vectors Integration
- Context Derivation
- And so much more

---

## Data Cleaning Options

- **Case Normalization**
- **Removing Stop Words**
- **Removing Punctuations or Special Symbols**
- **Lemmatization and Stemming** (word normalization)

---

## Parts of Speech Tagging

A Part-Of-Speech Tagger (POS Tagger) is a piece of software that reads text in some language and assigns parts of speech to each word (and other token), such as noun, verb, adjective, etc., although generally computational applications use more fine-grained POS tags like ‘noun-plural.

---

## Entity Detection

Named entity recognition (NER), also known as entity chunking/extraction, is a popular technique used in information extraction to identify and segment the named entities and classify or categorize them under various predefined classes.

---

## Dependency Parsing

Syntactic Parsing or Dependency Parsing is the task of recognizing a sentence and assigning a syntactic structure to it. The most widely used syntactic structure is the **parse tree**, which can be generated using some parsing algorithms. These parse trees are useful in various applications like grammar checking or more importantly, it plays a critical role in the semantic analysis stage.

---

## Noun Phrases

Noun phrases are part of speech patterns that include a noun. They can also include whatever other parts of speech make grammatical sense, and can include multiple nouns. Some common noun phrase patterns are:

- Noun
- Noun-Noun..… -Noun
- Adjective(s)-Noun
- Verb-(Adjectives-)Noun

---

## Words-to-Vectors Integration

Computers interact with humans in programming languages which are unambiguous, precise, and often structured. However, natural (human) language has a lot of ambiguity. There are multiple words with the same meaning (synonyms), words with multiple meanings (polysemy) some of which are entirely opposite in nature (auto-antonyms), and words which behave differently when used as noun and verb. These words make sense contextually in natural language which humans can comprehend and distinguish easily, but machines can’t. That’s what makes NLP one of the most difficult and interesting tasks in AI.

**Word2Vec** is a group of models which helps derive relations between a word and its contextual words.

---

## Case Normalization

Normalization is a process that converts a list of words to a more uniform sequence. For example, converting all words to lowercase will simplify the searching process.

---

## Removing Stop Words

A stop word is a commonly used word (such as “the”, “a”, “an”, “in”) that a search engine has been programmed to ignore, both when indexing entries for searching and when retrieving them as the result of a search query.

To check the list of stopwords, you can type the following commands in the Python shell.

```python
import nltk
from nltk.corpus import stopwords
print(stopwords.words('english'))
```

---

## Stemming

Stemming is the process of reducing inflection in words to their root forms such as mapping a group of words to the same stem even if the stem itself is not a valid word in the Language.

- Playing → Play
- Plays → Play
- Played → Play

---

## Lemmatisation

Lemmatisation (or lemmatization) in linguistics is the process of grouping together the inflected forms of a word so they can be analysed as a single item, identified by the word’s lemma, or dictionary form.

- is, am, are → be
- Example: "the boy's cars are different colors" → "the boy car be differ color"

---

## What will be covered in this blog

- Introduction of NLP and Spam detection using sklearn
- Reading Text and PDF files in Python
- Tokenization
- Parts of Speech Tagging
- Word-to-Vectors
- Then real-world practical examples

---

## Bag of Words – The Simplest Word Embedding Technique

```python
doc1 = "I am high"
doc2 = "Yes I am high"
doc3 = "I am kidding"
```

By comparing the vectors, we see that some words are common.

---

## Bag of Words and Tf-idf

[https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html)

**tf–idf for “Term Frequency times Inverse Document Frequency”**

---

## Let’s start now the coding part

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('spam.tsv', sep='\t')
df.head()
```

| label | message                                      | length | punct |
|-------|----------------------------------------------|--------|-------|
| ham   | Go until jurong point, crazy.. Available only … | 111    | 9     |
| ham   | Ok lar… Joking wif u oni…                    | 29     | 6     |
| spam  | Free entry in 2 a wkly comp to win FA Cup fina… | 155    | 6     |
| ham   | U dun say so early hor… U c already then say… | 49     | 6     |
| ham   | Nah I don’t think he goes to usf, he lives aro… | 61     | 2     |

---

## Data Cleaning and Balancing

```python
df.isnull().sum()
```

```
label      0
message    0
length     0
punct      0
dtype: int64
```

```python
len(df)
```

```
5572
```

```python
df['label'].value_counts()
```

```
ham     4825
spam     747
Name: label, dtype: int64
```

Balancing the data:

```python
ham = df[df['label']=='ham']
spam = df[df['label']=='spam']
ham = ham.sample(spam.shape[0])
data = ham.append(spam, ignore_index=True)
```

---

## Exploratory Data Analysis

```python
plt.hist(data[data['label']=='ham']['length'], bins = 100, alpha = 0.7, label='Ham')
plt.hist(data[data['label']=='spam']['length'], bins = 100, alpha = 0.7, label='Spam')
plt.xlabel('length of messages')
plt.ylabel('Frequency')
plt.legend()
plt.xlim(0,300)
plt.show()
```

```python
plt.hist(data[data['label']=='ham']['punct'], bins = 100, alpha = 0.7, label='Ham')
plt.hist(data[data['label']=='spam']['punct'], bins = 100, alpha = 0.7, label='Spam')
plt.xlabel('punctuations')
plt.ylabel('Frequency')
plt.legend()
plt.xlim(0,30)
plt.show()
```

---

## Data Preparation

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

data.head()
```

| label | message                                      | length | punct |
|-------|----------------------------------------------|--------|-------|
| ham   | No problem with the renewal. I.ll do it right … | 79     | 3     |
| ham   | Good afternoon, my love! How goes that day ? I… | 160    | 5     |
| ham   | Can come my room but cannot come my house cos … | 77     | 6     |
| ham   | I can send you a pic if you like             | 35     | 2     |
| ham   | I am on the way to ur home                   | 26     | 0     |

---

## Word2Vec Implementation

```python
import gensim
from nltk.tokenize import word_tokenize
import numpy as np

embedding_dim = 100
text = data['message']

Text = []
for i in range(data.shape[0]):
    text1 = word_tokenize(text[i])
    Text = text1 + Text

model = gensim.models.Word2Vec(sentences=[Text], size=embedding_dim, workers=4, min_count=1)
words = list(model.wv.vocab)
```

---

## Word2Vec Function

```python
def word_2_vec(x):
    t1 = word_tokenize(x)
    model[t1]
    v = list(map(lambda y: sum(y)/len(y), zip(*model[t1])))
    a = np.array(v)
    return a.reshape(1,-1)
```

---

## Applying Word2Vec to Each Text Message

```python
data['vec'] = data['message'].apply(lambda x: word_2_vec(x))
data.head()
```

| label | message                                      | length | punct | vec                                                                 |
|-------|----------------------------------------------|--------|-------|---------------------------------------------------------------------|
| ham   | No problem with the renewal. I.ll do it right … | 79     | 3     | [[-0.028193846775037754, -0.000255275213728762…]                    |
| ham   | Good afternoon, my love! How goes that day ? I… | 160    | 5     | [[-0.03519534362069527, -0.0009830875404938859…]                    |
| ham   | Can come my room but cannot come my house cos … | 77     | 6     | [[-0.004332678098257424, -0.000847288884769012…]                    |
| ham   | I can send you a pic if you like             | 35     | 2     | [[-0.04251667247577147, -0.002708293984390118, …]                   |
| ham   | I am on the way to ur home                   | 26     | 0     | [[-0.040913782135248766, 0.0017535838996991515…]                    |

---

## Feature Vector Conversion

```python
w_vec = np.concatenate(data['vec'].to_numpy(), axis=0)
w_vec.shape
```

```
(1494, 100)
```

```python
word_vec = pd.DataFrame(w_vec)
word_vec.head()
```

| 0         | 1         | 2         | 3         | 4         | ... | 99        |
|-----------|-----------|-----------|-----------|-----------|-----|-----------|
| -0.028194 | -0.000255 | -0.005906 | 0.001543  | 0.028490  | ... | 0.007982  |
| -0.035195 | -0.000983 | -0.006663 | 0.002052  | 0.033874  | ... | 0.012224  |
| -0.004333 | -0.000847 | -0.001710 | 0.000431  | 0.003813  | ... | 0.001190  |
| -0.042517 | -0.002708 | -0.007234 | 0.001933  | 0.039980  | ... | 0.014543  |
| -0.040914 | 0.001754  | -0.009190 | 0.003661  | 0.040343  | ... | 0.014350  |

---

## Model Training and Evaluation

### Support Vector Machine (SVM)

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

Parameter_svc = [{'cache_size': [300, 400, 200], 'tol': [0.0011, 0.002, 0.003],
                'kernel': ['rbf', 'poly'],
                'degree': [3, 4, 5]}]

clf_svc = GridSearchCV(SVC(), Parameter_svc, scoring='accuracy', verbose=2, cv=5)
clf_svc.fit(X_train, y_train)
print(clf_svc.best_params_)
```

**Source:** [https://kgptalkie.com/nlp-tutorial-spam-text-message-classification-using-nlp](https://kgptalkie.com/nlp-tutorial-spam-text-message-classification-using-nlp)

---

### LGBMClassifier

```python
from lightgbm import LGBMClassifier

Parameter_lgbm = [{'num_leaves': [31, 40, 50],
                 'max_depth': [3, 4, 5, 6],
                 'learning_rate': [0.1, 0.05, 0.2, 0.15],
                 'n_estimators': [700]}]

clf_lgbm = GridSearchCV(LGBMClassifier(), Parameter_lgbm, scoring='accuracy', verbose=2, cv=5)
clf_lgbm.fit(X_train, y_train)
print(clf_lgbm.best_params_)
```

**Source:** [https://kgptalkie.com/nlp-tutorial-spam-text-message-classification-using-nlp](https://kgptalkie.com/nlp-tutorial-spam-text-message-classification-using-nlp)

---

### Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier

RF_Cl = RandomForestClassifier(n_estimators=900)
RF_Cl.fit(X_train, y_train)
y_pred = RF_Cl.predict(X_test)
accuracy_score(y_pred, y_test)
```

**Source:** [https://kgptalkie.com/nlp-tutorial-spam-text-message-classification-using-nlp](https://kgptalkie.com/nlp-tutorial-spam-text-message-classification-using-nlp)

---

## Classification using TFidf

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_train_vect.shape
```

```
(1045, 3596)
```

---

## Pipeline and Random Forest Classifier

```python
from sklearn.pipeline import Pipeline

clf_rf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1))])
clf_rf.fit(X_train, y_train)
y_pred = clf_rf.predict(X_test)
confusion_matrix(y_test, y_pred)
```

```
[[224,   1],
 [ 24, 200]]
```

**Source:** [https://kgptalkie.com/nlp-tutorial-spam-text-message-classification-using-nlp](https://kgptalkie.com/nlp-tutorial-spam-text-message-classification-using-nlp)

---

## Support Vector Machine with TFidf

```python
clf_svc = Pipeline([('tfidf', TfidfVectorizer()), ('clf', SVC(C=1000, gamma='auto'))])
clf_svc.fit(X_train, y_train)
y_pred = clf_svc.predict(X_test)
confusion_matrix(y_test, y_pred)
```

```
[[221,   4],
 [ 16, 208]]
```

**Source:** [https://kgptalkie.com/nlp-tutorial-spam-text-message-classification-using-nlp](https://kgptalkie.com/nlp-tutorial-spam-text-message-classification-using-nlp)

---

## Summary

From above results, we can conclude that **TFidf** performs better than **word embeddings**, since even without hyperparameter tuning, all models performed well on test data with accuracy above 93% in all models.