https://kgptalkie.com/feature-selection-based-on-univariate-anova-test-for-classification-machine-learning-kgp-talkie

# Feature Selection Based on Univariate (ANOVA) Test for Classification | Machine Learning | KGP Talkie

**Source:** https://kgptalkie.com/feature-selection-based-on-univariate-anova-test-for-classification-machine-learning-kgp-talkie

**Published by:** KGP Talkie  
**Date:** 11 August 2020

## Watch Full Playlist
https://www.youtube.com/playlist?list=PLc2rvfiptPSQYzmDIFuq2PqN2n28ZjxDH

## What is Univariate (ANOVA) Test

The elimination process aims to reduce the size of the input feature set and at the same time to retain the class discriminatory information for classification problems.

- **F-test**: Any statistical test where the test statistic has an F-distribution under the null hypothesis.
- **ANOVA**: A collection of statistical models used to analyze differences among group means in a sample.

The F-test is used for comparing the factors of the total deviation. For example, in one-way, or single-factor ANOVA, statistical significance is tested for by comparing the F test statistic.

---

## Classification Problem

### Importing Required Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_classif, f_regression
from sklearn.feature_selection import SelectKBest, SelectPercentile
```

### Download Data Files

https://github.com/laxmimerit/Data-Files-for-Feature-Selection

```python
data = pd.read_csv('train.csv', nrows=20000)
data.head()
```

| ID | var3 | var15 | imp_ent_var16_ult1 | ... | TARGET |
|----|------|-------|--------------------|-----|--------|
| 0  | 1    | 2     | 23                 | ... | 0      |
| 1  | 3    | 2     | 34                 | ... | 0      |
| 2  | 4    | 2     | 23                 | ... | 0      |
| 3  | 8    | 2     | 37                 | ... | 0      |
| 4  | 10   | 2     | 39                 | ... | 0      |

```python
X = data.drop('TARGET', axis=1)
y = data['TARGET']
X.shape, y.shape
```

Output:
```
((20000, 370), (20000,))
```

### Train, Test, and Split Data

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
```

---

## Remove Constant, Quasi Constant, and Correlated Features

### Remove Constant and Quasi Constant Features

```python
constant_filter = VarianceThreshold(threshold=0.01)
constant_filter.fit(X_train)
X_train_filter = constant_filter.transform(X_train)
X_test_filter = constant_filter.transform(X_test)
```

```python
X_train_filter.shape, X_test_filter.shape
```

Output:
```
((16000, 245), (4000, 245))
```

### Remove Duplicate Features

```python
X_train_T = X_train_filter.T
X_test_T = X_test_filter.T
X_train_T = pd.DataFrame(X_train_T)
X_test_T = pd.DataFrame(X_test_T)
```

```python
X_train_T.duplicated().sum()
```

Output:
```
18
```

```python
duplicated_features = X_train_T.duplicated()
features_to_keep = [not index for index in duplicated_features]
X_train_unique = X_train_T[features_to_keep].T
X_test_unique = X_test_T[features_to_keep].T
```

```python
X_train_unique.shape, X_train.shape
```

Output:
```
((16000, 227), (16000, 370))
```

---

## Now Do F-Test

```python
sel = f_classif(X_train_unique, y_train)
sel
```

Output:
```
(array([3.42911520e-01, 1.22929093e+00, 1.61291330e+02, ...]), 
 array([5.58161700e-01, 2.67561647e-01, 8.89333290e-37, ...]))
```

### Plot P-Values

```python
p_values = pd.Series(sel[1])
p_values.index = X_train_unique.columns
p_values.sort_values(ascending=True, inplace=True)
p_values.plot.bar(figsize=(16, 5))
plt.title('pvalues with respect to features')
plt.show()
```

```python
p_values = p_values[p_values < 0.05]
p_values.index
```

Output:
```
Int64Index([ 40, 182,  86,  22, 101,  51,   2, 127,  49,  91, ...], dtype='int64')
```

```python
X_train_p = X_train_unique[p_values.index]
X_test_p = X_test_unique[p_values.index]
```

---

## Build the Classifiers and Compare the Performance

### Random Forest Classifier

```python
def run_randomForest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy: ', accuracy_score(y_test, y_pred))
```

### Accuracy with Filtered Data

```python
%%time
run_randomForest(X_train_p, X_test_p, y_train, y_test)
```

Output:
```
Accuracy:  0.953
Wall time: 814 ms
```

### Accuracy with Original Data

```python
%%time
run_randomForest(X_train, X_test, y_train, y_test)
```

Output:
```
Accuracy:  0.9585
Wall time: 1.49 s
```