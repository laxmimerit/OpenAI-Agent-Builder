https://kgptalkie.com/star-rating-prediction

# Star Rating Prediction

**Published by** pasqualebrownlow **on** 28 August 2020  
28 August 2020

## Star Rating Prediction of Amazon Products Reviews

### Objective

In this notebook, we are going to predict the Ratings of Amazon products reviews by the help of given `reviewText` column.

---

## Natural Language Processing (NLP)

Natural Language Processing (NLP) is a sub-field of artificial intelligence that deals with understanding and processing human language. In light of new advancements in **machine learning**, many organizations have begun applying natural language processing for translation, chatbots, and candidate filtering.

Machine learning algorithms cannot work with raw text directly. Rather, the text must be converted into vectors of numbers. Then we use **TF-IDF vectorizer** approach.

### TF-IDF

TF-IDF is a technique used for natural language processing that transforms text to feature vectors that can be used as input to the estimator.

---

## Intro to Pandas

Pandas is a column-oriented data analysis API. It’s a great tool for handling and analyzing input data, and many ML frameworks support **pandas** data structures as inputs. Although a comprehensive introduction to the **pandas** API would span many pages, the core concepts are fairly straightforward, and we will present them below. For a more complete **reference**, the **pandas** docs site contains extensive documentation and many tutorials.

---

## Intro to Numpy

Numpy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays. For a more complete **reference**, the **numpy** docs site contains extensive documentation and many tutorials.

---

## Installation

Firstly install the **pandas**, **numpy**, **scikit-learn** library.

```python
!pip install pandas
!pip install numpy
!pip install scikit-learn
```

```python
import pandas as pd
import numpy as np
```

---

## Dataset

Dataset is available [here](https://kgptalkie.com/star-rating-prediction)

```python
df = pd.read_csv('https://raw.githubusercontent.com/laxmimerit/Amazon-Musical-Reviews-Rating-Dataset/master/Musical_instruments_reviews.csv', usecols = ['reviewText', 'overall'])
```

---

## Data Sampling

```python
df.sample(5)
```

| reviewText                                                                 | overall |
|----------------------------------------------------------------------------|---------|
| Cheap and good. Just what I needed. No issue…                              | 4.0     |
| It sounds like it’s Behringer. Very fake, che…                            | 3.0     |
| I already had the nickel finished version (whi…                          | 5.0     |
| Well… it’s not really too expensive, but it …                           | 1.0     |
| The mic stand pick holder is a great way to ke…                         | 5.0     |

```python
df['overall'].value_counts()
```

```
5.0    6938
4.0    2084
3.0     772
2.0     250
1.0     217
Name: overall, dtype: int64
```

```python
df1 = pd.DataFrame()
for val in df['overall'].unique():
  temp = df[df['overall']==val].sample(217)
  df1 = df1.append(temp, ignore_index = True)
```

---

## Text Preprocessing

In natural language processing (NLP), text preprocessing is the practice of cleaning and preparing text data. **NLTK** and **re** are common Python libraries used to handle many text preprocessing tasks.

**preprocess_kgptalkie** python package is prepared by [Kgptalkie](https://kgptalkie.com/star-rating-prediction)

These are the some dependencies that you have to install before using this **preprocess_kgptalkie** package.

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

### `get_clean` Function

Defining `get_clean` function which is taking argument as ‘Reviews’ column then after perform some steps:

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

```python
df['reviewText'] = df['reviewText'].apply(lambda x: get_clean(x))
```

```python
df.head()
```

| reviewText                                                                 | overall |
|----------------------------------------------------------------------------|---------|
| not much to write about here but it does exact…                          | 5.0     |
| the product does exactly as it should and is q…                          | 5.0     |
| the primary job of this device is to block the…                          | 5.0     |
| nice windscreen protects my mxl mic and preven…                          | 5.0     |
| this pop filter is great it looks and performs…                          | 5.0     |

---

## TF-IDF Vectorizer

Some semantic information is preserved as uncommon words are given more importance than common words in **TF-IDF**.

**E.g.** 'She is beautiful', Here 'beautiful' will have more importance than 'she' or 'is'.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
```

```python
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,5), analyzer='char')
X = tfidf.fit_transform(df['reviewText'])
y = df['overall']
```

```python
X.shape, y.shape
```

```
((10261, 20000), (10261,))
```

Here, splitting the dataset into x and y column having 20% is for testing and 80% for training purpose.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

```python
X_train.shape
```

```
(8208, 20000)
```

---

## Support Vector Machine

### Definition

**SVM** is a supervised machine learning algorithm which can be used for classification or regression problems. It uses a technique called the kernel trick to transform your data and then based on these transformations it finds an optimal boundary between the possible outputs.

The objective of a **Linear SVC** (Support Vector Classifier) is to fit to the data you provide, returning a “best fit” hyperplane that divides, or categorizes, your data. From there, after getting the hyperplane, you can then feed some features to your classifier to see what the “predicted” class is.

```python
clf = LinearSVC(C = 20, class_weight='balanced')
clf.fit(X_train, y_train)
```

```
C:\Users\md  ezajul hassan\AppData\Local\Programs\Python\Python37\lib\site-packages\sklearn\svm\_base.py:977: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
```

```python
y_pred = clf.predict(X_test)
```

### Classification Report

The **classification report** shows a representation of the main classification metrics on a per-class basis. This gives a deeper intuition of the classifier behavior over global accuracy which can mask functional weaknesses in one class of a multiclass problem.

```python
print(classification_report(y_test, y_pred))
```

```
              precision    recall  f1-score   support

           1.0       0.31      0.21      0.25        39
           2.0       0.18      0.11      0.13        55
           3.0       0.23      0.27      0.25       134
           4.0       0.34      0.33      0.34       451
           5.0       0.77      0.78      0.78      1374

    accuracy                           0.62      2053
   macro avg       0.37      0.34      0.35      2053
weighted avg       0.61      0.62      0.62      2053
```

---

## Example Predictions

```python
x = 'this product is really bad. i do not like it'
x = get_clean(x)
vec = tfidf.transform([x])
clf.predict(vec)
```

```
array([1.])
```

```python
x = 'this product is really good. thanks a lot for speedy delivery'
x = get_clean(x)
vec = tfidf.transform([x])
clf.predict(vec)
```

```
array([5.])
```

---

## Conclusion

1. Firstly, We have loaded the Amazon musical reviews rating dataset using **pandas** dataframe.
2. Then define `get_clean()` function and removed unwanted emails, URLs, Html tags and special character.
3. Convert the text into vectors with the help of the **TF-IDF Vectorizer**.
4. After that use a linear vector machine classifier algorithm.
5. Finally, we have fit the model on the **LinearSVC** classifier for categorical classification and predict the rating on real data.

By the help of these steps, we got **62% accuracy**.

**Source:** https://kgptalkie.com/star-rating-prediction