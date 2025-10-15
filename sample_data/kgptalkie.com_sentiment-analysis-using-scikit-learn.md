https://kgptalkie.com/sentiment-analysis-using-scikit-learn

# Sentiment Analysis Using Scikit-learn

## Published by
pasqualebrownlow

## Date
25 August 2020

---

### Objective

In this notebook, we are going to perform a binary classification, i.e., we will classify the sentiment as **positive** or **negative** according to the `Reviews` column data of the IMDB dataset. We will use **TFIDF** for text data vectorization and **Linear Support Vector Machine** for classification.

---

## Natural Language Processing (NLP)

Natural Language Processing (NLP) is a sub-field of artificial intelligence that deals with understanding and processing human language. In light of new advancements in **machine learning**, many organizations have begun applying natural language processing for translation, chatbots, and candidate filtering.

Machine learning algorithms cannot work with raw text directly. Rather, the text must be converted into vectors of numbers. Then we use **TF-IDF** vectorizer approach.

---

## TF-IDF

TF-IDF is a technique used for natural language processing that transforms text to feature vectors that can be used as input to the estimator.

---

## Intro to Pandas

Pandas is a column-oriented data analysis API. It’s a great tool for handling and analyzing input data, and many ML frameworks support **pandas** data structures as inputs. Although a comprehensive introduction to the **pandas** API would span many pages, the core concepts are fairly straightforward, and we will present them below. For a more complete reference, the **pandas** docs site contains extensive documentation and many tutorials.

---

## Intro to Numpy

Numpy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

---

## Installing Libraries

Firstly, install the **pandas**, **numpy**, and **scikit-learn** library.

```python
!pip install pandas
!pip install numpy
!pip install scikit-learn
```

---

## Let’s Get Started

```python
import pandas as pd
import numpy as np
```

The dataset is available here:  
**Source:** https://kgptalkie.com/sentiment-analysis-using-scikit-learn

---

## Cloning the Dataset

```python
!git clone https://github.com/laxmimerit/IMDB-Movie-Reviews-Large-Dataset-50k.git
```

Cloning into 'IMDB-Movie-Reviews-Large-Dataset-50k'...

---

## Reading an Excel File into a Pandas DataFrame

```python
df = pd.read_excel('IMDB-Movie-Reviews-Large-Dataset-50k/train.xlsx')
```

---

## TF-IDF Vectorization

Some semantic information is preserved as uncommon words are given more importance than common words in TF-IDF.  
E.g., 'She is beautiful', here 'beautiful' will have more importance than 'she' or 'is'.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# Displaying top 5 rows of our dataset
df.head()
```

| Reviews | Sentiment |
|--------|----------|
| When I first tuned in on this morning news, I … | neg |
| Mere thoughts of “Going Overboard” (aka “Babes… | neg |
| Why does this movie fall WELL below standards?… | neg |
| Wow and I thought that any Steven Segal movie … | neg |
| The story is seen before, but that does’n matt… | neg |

---

## Text Preprocessing

In natural language processing (NLP), text preprocessing is the practice of cleaning and preparing text data. **NLTK** and **re** are common Python libraries used to handle many text preprocessing tasks.

The **preprocess_kgptalkie** python package is prepared by **Kgptalkie**.

These are some dependencies that you have to install before using this **preprocess_kgptalkie** package.

```python
!pip install spacy==2.2.3
!python -m spacy download en_core_web_sm
!pip install beautifulsoup4==4.9.1
!pip install textblob==0.15.3
```

Importing **preprocess_kgptalkie** python package and also regular expression (**re**).

```python
import preprocess_kgptalkie as ps
import re
```

---

## Defining `get_clean` Function

```python
def get_clean(x):
    x = str(x).lower().replace('\\', '').replace('_', ' ')
    x = ps.cont_exp(x)
    x = ps.remove_emails(x)
    x = ps.remove_urls(x)
    x = ps.remove_html_tags(x)
    x = ps.remove_accented_chars(x)
    x = ps.remove_special_chars(x)
    x = re.sub("(.)\\1{2,}", "\\1", x)
    return x
```

**Source:** https://kgptalkie.com/sentiment-analysis-using-scikit-learn

---

## Applying `get_clean` Function

```python
df['Reviews'] = df['Reviews'].apply(lambda x: get_clean(x))
df.head()
```

| Reviews | Sentiment |
|--------|----------|
| when i first tuned in on this morning news i t… | neg |
| mere thoughts of going overboard aka babes aho… | neg |
| why does this movie fall well below standards … | neg |
| wow and i thought that any steven segal movie … | neg |
| the story is seen before but that doesn matter… | neg |

---

## TF-IDF Vectorizer

```python
tfidf = TfidfVectorizer(max_features=5000)
X = df['Reviews']
y = df['Sentiment']

X = tfidf.fit_transform(X)
X
```

```
<25000x5000 sparse matrix of type '<class 'numpy.float64'>'
	with 2843804 stored elements in Compressed Sparse Row format>
```

---

## Splitting the Dataset

Here, splitting the dataset into x and y column having 20% is for testing and 80% for training purposes.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

---

## Support Vector Machine (SVM)

### Definition

**SVM** is a supervised machine learning algorithm that can be used for classification or regression problems. It uses a technique called the kernel trick to transform your data and then based on these transformations, it finds an optimal boundary between the possible outputs.

The objective of a **Linear SVC** (Support Vector Classifier) is to fit the data you provide, returning a “best fit” hyperplane that divides, or categorizes your data. From there, after getting the hyperplane, you can then feed some features to your classifier to see what the “predicted” class is.

---

## Training the Model

```python
clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

---

## Classification Report

The **classification report** shows a representation of the main classification metrics on a per-class basis. This gives a deeper intuition of the classifier behavior over global accuracy which can mask functional weaknesses in one class of a multiclass problem.

```python
print(classification_report(y_test, y_pred))
```

```
              precision    recall  f1-score   support

         neg       0.87      0.87      0.87      2480
         pos       0.87      0.88      0.88      2520

    accuracy                           0.87      5000
   macro avg       0.87      0.87      0.87      5000
weighted avg       0.87      0.87      0.87      5000
```

**Source:** https://kgptalkie.com/sentiment-analysis-using-scikit-learn

---

## Predicting New Data

```python
x = 'this movie is really good. thanks a lot for making it'
x = get_clean(x)
vec = tfidf.transform([x])
vec.shape
```

```
(1, 5000)
```

```python
clf.predict(vec)
```

```
array(['pos'], dtype=object)
```

---

## Python `pickle` Module

The **Python pickle** module is used for serializing and de-serializing Python object structures. The process to converts any kind of Python objects (list, dict, etc.) into byte streams (0s and 1s) is called **pickling** or **serialization** or **flattening** or **marshalling**. We can convert the byte stream (generated through pickling) back into Python objects by a process called as **unpickling**.

```python
import pickle
pickle.dump(clf, open('model', 'wb'))
pickle.dump(tfidf, open('tfidf', 'wb'))
```

---

## Conclusions

- Firstly, we have loaded the IMDB movie reviews dataset using the **pandas** dataframe.
- Then, we defined `get_clean()` function and removed unwanted emails, URLs, HTML tags, and special characters.
- Convert the text into vectors with the help of the **TF-IDF** Vectorizer.
- After that, we used a linear vector machine classifier algorithm.
- We have fit the model on **LinearSVC** classifier for binary classification and predicted the sentiment (i.e., positive or negative) on real data.
- Lastly, we dumped the `clf` and **TF-IDF** model with the help of the **pickle** library. In other words, it’s the process of converting a Python object into a byte stream to store it in a file/database, maintain program state across sessions, or transport data over the network.

**Source:** https://kgptalkie.com/sentiment-analysis-using-scikit-learn