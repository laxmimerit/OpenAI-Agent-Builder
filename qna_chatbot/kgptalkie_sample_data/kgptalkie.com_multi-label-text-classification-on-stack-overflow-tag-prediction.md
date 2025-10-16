[https://kgptalkie.com/multi-label-text-classification-on-stack-overflow-tag-prediction](https://kgptalkie.com/multi-label-text-classification-on-stack-overflow-tag-prediction)

# Multi-Label Text Classification on Stack Overflow Tag Prediction

**Published by**  
crystlefroggatt  
**on**  
25 August 2020

## Multi-Label Text Classification

In this notebook, we will use the dataset “StackSample: 10% of Stack Overflow Q&A” and use the questions and the tags data. We will be developing a text classification model that analyzes a textual comment and predicts multiple labels associated with the questions. We will implement a tag suggestion system using **Multi-Label Text Classification**, which is a subset of multiple output models.

### Text Preprocessing

Text preprocessing is performed on the text data, and the cleaned data is loaded for text classification.

### Text Vectorization and Modeling

We will implement text vectorization on text data, encode the tag labels using `MultilabelBinarizer`, and model classical classifiers (SGC classifier, Multi-Nomial Naive Bayes Classifier, Random Forest Classifier, etc.) for modeling and compare the results.

---

## Classification in Machine Learning

In **machine learning**, classification is a type of supervised learning. **Classification** refers to a predictive modeling problem where a class label is predicted for a given input sample. It specifies the class to which a data point belongs and is best used when the output has finite and discrete values.

There are 4 types of classification tasks:

1. **Binary Classification**  
   Predicting one of two classes.  
   - Email spam detection (spam or not)  
   - Churn prediction (churn or not)  
   - Conversion prediction (buy or not)

2. **Multi-class Classification**  
   A task with more than two classes where each sample is assigned to one and only one class.  
   - A fruit can be either an apple or a pear but not both at the same time.

3. **Multi-Label Classification**  
   Refers to classification tasks with two or more class labels, where one or more class labels may be predicted for each sample.  
   - Example: Photo classification with multiple objects like “bicycle,” “apple,” “person,” etc.

4. **Imbalanced Classification**  
   Refers to classification tasks where the number of examples in each class is unequally distributed.  
   - Fraud detection  
   - Outlier detection  
   - **Source**: [cse-iitk](https://kgptalkie.com/multi-label-text-classification-on-stack-overflow-tag-prediction)

---

## Notebook Setup

### Import Libraries

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
```

**Source**: [https://kgptalkie.com/multi-label-text-classification-on-stack-overflow-tag-prediction](https://kgptalkie.com/multi-label-text-classification-on-stack-overflow-tag-prediction)

### Load Data

```python
df = pd.read_csv('https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/stackoverflow.csv', index_col=0)
```

### Display First Few Rows

```python
df.head()
```

| Text | Tags |
|------|------|
| aspnet site maps has anyone got experience cre… | ['sql', 'asp.net'] |
| adding scripting functionality to net applicat… | ['c#', '.net'] |
| should i use nested classes in this case i am … | ['c++'] |
| homegrown consumption of web services i have b… | ['.net'] |
| automatically update version number i would li… | ['c#'] |

### Clean Tags

```python
import ast
df['Tags'] = df['Tags'].apply(lambda x: ast.literal_eval(x))
```

---

## Encoding Categorical Features

### LabelEncoder

- Used for transforming non-numerical labels to numerical labels.
- Encodes labels with values between 0 and `n_classes - 1`.
- Suitable for binary columns (e.g., YES/NO, Male/Female).

### OneHotEncoder

- Converts categorical variables into a form that can be provided to ML algorithms.
- Handles input string categorical and numeric data.
- Avoids **variable-trap** using the `drop` parameter.

### get_dummies

- Pandas method to create dummy variables for categorical features.
- Uses `drop_first=True` to avoid multicollinearity.

### MultiLabel Binarize

- Encodes multiple labels per instance.
- Example:

```python
multilabel = MultiLabelBinarizer()
y = multilabel.fit_transform(df['Tags'])
```

---

## Text Vectorization

### Word Embeddings

- Maps words or phrases to real numbers for NLP tasks.
- **TF-IDF** is a statistical measure that evaluates word relevance in a document.

### TF-IDF Vectorizer

```python
tfidf = TfidfVectorizer(analyzer='word', max_features=10000, ngram_range=(1,3), stop_words='english')
X = tfidf.fit_transform(df['Text'])
```

### Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

---

## Build Model

### Classifiers

```python
sgd = SGDClassifier()
lr = LogisticRegression(solver='lbfgs')
svc = LinearSVC()
```

### Metrics for Multi-Label Classification

- **Hamming Loss**: Average fraction of incorrect labels.  
  $$
  \text{Hamming loss} = \frac{TP + TN}{T}
  $$
  where $ T = TP + TN + FP + FN $ (total number of labels).

- **Jaccard Similarity**:  
  $$
  \text{Jaccard Score} = \frac{TP}{TP + FP + FN} \quad \text{if } TN \neq 1
  $$

```python
def j_score(y_true, y_pred):
    jaccard = np.minimum(y_true, y_pred).sum(axis=1) / np.maximum(y_true, y_pred).sum(axis=1)
    return jaccard.mean() * 100
```

**Source**: [https://kgptalkie.com/multi-label-text-classification-on-stack-overflow-tag-prediction](https://kgptalkie.com/multi-label-text-classification-on-stack-overflow-tag-prediction)

---

## OneVsRest Classifier

- Strategy for multi-class/multi-label classification.
- Fits one classifier per class.

```python
for classifier in [LinearSVC(C=1.5, penalty='l1', dual=False)]:
    clf = OneVsRestClassifier(classifier)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print_score(y_pred, classifier)

for classifier in [sgd, lr, svc]:
    clf = OneVsRestClassifier(classifier)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print_score(y_pred, classifier)
```

---

## Model Test with Real Data

```python
x = ['how to write ml code in python and java i have data but do not know how to do it']
xt = tfidf.transform(x)
clf.predict(xt)
multilabel.inverse_transform(clf.predict(xt))
```

---

## Conclusion

- Loaded and preprocessed the dataset using pandas.
- Evaluated string tags using the `ast` module and encoded them with `MultilabelBinarizer`.
- Performed text vectorization using `TfidfVectorizer`.
- Fitted models on classifiers like `LinearSVC`, `SGDClassifier`, and `LogisticRegression` for multi-label classification.
- Predicted output on real data.

For better accuracy, consider using **RNN**, **LSTM**, or **bi-directional LSTM** for multi-label text classification.