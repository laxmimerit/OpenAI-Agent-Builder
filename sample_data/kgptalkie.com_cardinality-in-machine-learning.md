https://kgptalkie.com/cardinality-in-machine-learning

# Feature Engineering Series Tutorial 2: Cardinality in Machine Learning

Published by  
georgiannacambel  
on  
29 September 2020

## What is Cardinality?

Cardinality refers to the number of possible values that a feature can assume. For example, the variable “US State” is one that has 50 possible values. The binary features, of course, could only assume one of two values (0 or 1).

The values of a categorical variable are selected from a group of categories, also called labels. For example, in the variable **gender**, the categories or labels are male and female, whereas in the variable **city**, the labels can be London, Manchester, Brighton, and so on.

Different categorical variables contain different numbers of labels or categories. The variable **gender** contains only 2 labels, but a variable like **city** or **postcode**, can contain a huge number of different labels.

The number of different labels within a categorical variable is known as **cardinality**. A high number of labels within a variable is known as **high cardinality**.

## Are Multiple Labels in a Categorical Variable a Problem?

High cardinality may pose the following problems:

- Variables with too many labels tend to dominate over those with only a few labels, particularly in **Tree based** algorithms.
- A big number of labels within a variable may introduce noise with little, if any, information, therefore making **machine learning** models prone to over-fit.
- Some of the labels may only be present in the training data set, but not in the test set, therefore machine learning algorithms may over-fit to the training set.
- Contrarily, some labels may appear only in the test set, therefore leaving the machine learning algorithms unable to perform a calculation over the new (unseen) observation.
- In particular, **tree methods can be biased towards variables with lots of labels** (variables with high cardinality). Thus, their performance may be affected by high cardinality.

## What We Will Cover

In this Blog:

- We will:
  - Learn how to quantify cardinality
  - See examples of high and low cardinality variables
  - Understand the effect of cardinality while preparing train and test sets
  - See the effect of cardinality on Machine Learning Model performance
- We will use the **Titanic dataset**.

## Getting Started

We will first import all the necessary libraries.

```python
# to read the dataset into a dataframe and perform operations on it
import pandas as pd

# to perform basic array operations
import numpy as np

# to build machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# to evaluate the models
from sklearn.metrics import roc_auc_score

# to separate data into train and test
from sklearn.model_selection import train_test_split
```

Now we will read the titanic dataset using `read_csv()`.

```python
data = pd.read_csv('https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/titanic.csv')
data.head()
```

Output:

```
   PassengerId  Survived  Pclass                          Name     Sex   Age  SibSp  Parch        Ticket     Fare   Cabin Embarked
0            1         0     3    Braund, Mr. Owen Harris    male  22.0      1      0     A/5 21171   7.2500      NaN        S
1            2         1     1  Cumings, Mrs. John Bradley (Florence Briggs Th…  female  38.0      1      0       PC 17599  71.2833     C85        C
2            3         1     3    Heikkinen, Miss. Laina    female  26.0      0      0  STON/O2. 3101282   7.9250      NaN        S
3            4         1     1  Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1      0         113803  53.1000     C123        S
4            5         0     3    Allen, Mr. William Henry    male  35.0      0      0         373450   8.0500      NaN        S
```

The categorical variables in this dataset are **Name**, **Sex**, **Ticket**, **Cabin**, and **Embarked**.

**Note:** `Ticket` and `Cabin` contain both letters and numbers, so they could be treated as *Mixed Variables*. For this demonstration, we will treat them as categorical variables.

## Inspecting Cardinality

```python
print('Number of categories in the variable Name: {}'.format(len(data.Name.unique())))
print('Number of categories in the variable Gender: {}'.format(len(data.Sex.unique())))
print('Number of categories in the variable Ticket: {}'.format(len(data.Ticket.unique())))
print('Number of categories in the variable Cabin: {}'.format(len(data.Cabin.unique())))
print('Number of categories in the variable Embarked: {}'.format(len(data.Embarked.unique())))
print('Total number of passengers in the Titanic: {}'.format(len(data)))
```

Output:

```
Number of categories in the variable Name: 891
Number of categories in the variable Gender: 2
Number of categories in the variable Ticket: 681
Number of categories in the variable Cabin: 148
Number of categories in the variable Embarked: 4
Total number of passengers in the Titanic: 891
```

While the variable **Sex** contains only 2 categories and **Embarked** contains 4 (low cardinality), the variables **Ticket**, **Name**, and **Cabin**, as expected, contain a huge number of different labels (high cardinality).

## Reducing Cardinality

To demonstrate the effect of high cardinality in train and test sets and machine learning performance, we will work with the variable **Cabin**. We will create a new variable with reduced cardinality.

```python
# let's capture the first letter of Cabin
data['Cabin_reduced'] = data['Cabin'].astype(str).str[0]
```

**Source:** https://kgptalkie.com/cardinality-in-machine-learning

```python
data[['Cabin', 'Cabin_reduced']].head()
```

Output:

```
   Cabin Cabin_reduced
0   NaN             n
1   C85             C
2   NaN             n
3  C123             C
4   NaN             n
```

Now let’s check the cardinality of **Cabin_reduced**. We reduced the number of different labels from 148 to 9.

```python
print('Number of categories in the variable Cabin: {}'.format(len(data.Cabin.unique())))
print('Number of categories in the variable Cabin_reduced: {}'.format(len(data.Cabin_reduced.unique())))
```

Output:

```
Number of categories in the variable Cabin: 148
Number of categories in the variable Cabin_reduced: 9
```

## Splitting Data into Train and Test Sets

```python
use_cols = ['Cabin', 'Cabin_reduced', 'Sex']

X_train, X_test, y_train, y_test = train_test_split(
    data[use_cols], 
    data['Survived'],  
    test_size=0.3,
    random_state=0
)

X_train.shape, X_test.shape
```

Output:

```
((623, 3), (268, 3))
```

## High Cardinality and Uneven Distribution

```python
unique_to_train_set = [x for x in X_train.Cabin.unique() if x not in X_test.Cabin.unique()]
len(unique_to_train_set)
```

Output:

```
100
```

```python
unique_to_test_set = [x for x in X_test.Cabin.unique() if x not in X_train.Cabin.unique()]
len(unique_to_test_set)
```

Output:

```
28
```

**Source:** https://kgptalkie.com/cardinality-in-machine-learning

```python
unique_to_train_set = [x for x in X_train['Cabin_reduced'].unique() if x not in X_test['Cabin_reduced'].unique()]
len(unique_to_train_set)
```

Output:

```
1
```

```python
unique_to_test_set = [x for x in X_test['Cabin_reduced'].unique() if x not in X_train['Cabin_reduced'].unique()]
len(unique_to_test_set)
```

Output:

```
0
```

## Effect of Cardinality on Machine Learning Model Performance

```python
import itertools
cabin_dict = {k: i for i, k in enumerate(X_train['Cabin'].unique(), 0)}
print(dict(itertools.islice(cabin_dict.items(), 100)))
```

**Source:** https://kgptalkie.com/cardinality-in-machine-learning

```python
X_train.loc[:, 'Cabin_mapped'] = X_train.loc[:, 'Cabin'].map(cabin_dict)
X_test.loc[:, 'Cabin_mapped'] = X_test.loc[:, 'Cabin'].map(cabin_dict)

X_train[['Cabin_mapped', 'Cabin']].head(10)
```

Output:

```
   Cabin_mapped   Cabin
857          0     E17
52           1     D33
386          2     NaN
124          3     D26
578          2     NaN
549          2     NaN
118          4  B58 B60
12           2     NaN
157          2     NaN
127          2     NaN
```

```python
cabin_dict = {k: i for i, k in enumerate(X_train['Cabin_reduced'].unique(), 0)}
X_train.loc[:, 'Cabin_reduced'] = X_train.loc[:, 'Cabin_reduced'].map(cabin_dict)
X_test.loc[:, 'Cabin_reduced'] = X_test.loc[:, 'Cabin_reduced'].map(cabin_dict)

X_train[['Cabin_reduced', 'Cabin']].head(20)
```

Output:

```
   Cabin_reduced   Cabin
857            0     E17
52             1     D33
386            2     NaN
124            1     D26
578            2     NaN
549            2     NaN
118            3  B58 B60
12             2     NaN
157            2     NaN
127            2     NaN
653            2     NaN
235            2     NaN
785            2     NaN
241            2     NaN
351            4   C128
862            1     D17
851            2     NaN
753            2     NaN
532            2     NaN
485            2     NaN
```

```python
X_train.loc[:, 'Sex'] = X_train.loc[:, 'Sex'].map({'male': 0, 'female': 1})
X_test.loc[:, 'Sex'] = X_test.loc[:, 'Sex'].map({'male': 0, 'female': 1})

X_train.Sex.head()
```

Output:

```
857    0
52     1
386    0
124    0
578    1
Name: Sex, dtype: int64
```

```python
X_train[['Cabin_mapped', 'Cabin_reduced', 'Sex']].isnull().sum()
```

Output:

```
Cabin_mapped     0
Cabin_reduced    0
Sex              0
dtype: int64
```

```python
X_test[['Cabin_mapped', 'Cabin_reduced', 'Sex']].isnull().sum()
```

Output:

```
Cabin_mapped     30
Cabin_reduced     0
Sex               0
dtype: int64
```

## Random Forests

```python
rf = RandomForestClassifier(n_estimators=200, random_state=39)
rf.fit(X_train[['Cabin_mapped', 'Sex']], y_train)
pred_train = rf.predict_proba(X_train[['Cabin_mapped', 'Sex']])
pred_test = rf.predict_proba(X_test[['Cabin_mapped', 'Sex']].fillna(0))

print('Train set')
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred_train[:,1])))
print('Test set')
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred_test[:,1])))
```

Output:

```
Train set
Random Forests roc-auc: 0.8617329342096702
Test set
Random Forests roc-auc: 0.8078571428571428
```

```python
rf = RandomForestClassifier(n_estimators=200, random_state=39)
rf.fit(X_train[['Cabin_reduced', 'Sex']], y_train)
pred_train = rf.predict_proba(X_train[['Cabin_reduced', 'Sex']])
pred_test = rf.predict_proba(X_test[['Cabin_reduced', 'Sex']])

print('Train set')
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred_train[:,1])))
print('Test set')
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred_test[:,1])))
```

Output:

```
Train set
Random Forests roc-auc: 0.8199550985878832
Test set
Random Forests roc-auc: 0.8332142857142857
```

## AdaBoost

```python
ada = AdaBoostClassifier(n_estimators=200, random_state=44)
ada.fit(X_train[['Cabin_mapped', 'Sex']], y_train)
pred_train = ada.predict_proba(X_train[['Cabin_mapped', 'Sex']])
pred_test = ada.predict_proba(X_test[['Cabin_mapped', 'Sex']].fillna(0))

print('Train set')
print('Adaboost roc-auc: {}'.format(roc_auc_score(y_train, pred_train[:,1])))
print('Test set')
print('Adaboost roc-auc: {}'.format(roc_auc_score(y_test, pred_test[:,1])))
```

Output:

```
Train set
Adaboost roc-auc: 0.8399546647578144
Test set
Adaboost roc-auc: 0.809375
```

```python
ada = AdaBoostClassifier(n_estimators=200, random_state=44)
ada.fit(X_train[['Cabin_reduced', 'Sex']], y_train)
pred_train = ada.predict_proba(X_train[['Cabin_reduced', 'Sex']])
pred_test = ada.predict_proba(X_test[['Cabin_reduced', 'Sex']].fillna(0))

print('Train set')
print('Adaboost roc-auc: {}'.format(roc_auc_score(y_train, pred_train[:,1])))
print('Test set')
print('Adaboost roc-auc: {}'.format(roc_auc_score(y_test, pred_test[:,1])))
```

Output:

```
Train set
Adaboost roc-auc: 0.8195863430294354
Test set
Adaboost roc-auc: 0.8332142857142857
```

## Logistic Regression

```python
logit = LogisticRegression(random_state=44, solver='lbfgs')
logit.fit(X_train[['Cabin_mapped', 'Sex']], y_train)
pred_train = logit.predict_proba(X_train[['Cabin_mapped', 'Sex']])
pred_test = logit.predict_proba(X_test[['Cabin_mapped', 'Sex']].fillna(0))

print('Train set')
print('Logistic regression roc-auc: {}'.format(roc_auc_score(y_train, pred_train[:,1])))
print('Test set')
print('Logistic regression roc-auc: {}'.format(roc_auc_score(y_test, pred_test[:,1])))
```

Output:

```
Train set
Logistic regression roc-auc: 0.8094564109238411
Test set
Logistic regression roc-auc: 0.7591071428571431
```

```python
logit = LogisticRegression(random_state=44, solver='lbfgs')
logit.fit(X_train[['Cabin_reduced', 'Sex']], y_train)
pred_train = logit.predict_proba(X_train[['Cabin_reduced', 'Sex']])
pred_test = logit.predict_proba(X_test[['Cabin_reduced', 'Sex']].fillna(0))

print('Train set')
print('Logistic regression roc-auc: {}'.format(roc_auc_score(y_train, pred_train[:,1])))
print('Test set')
print('Logistic regression roc-auc: {}'.format(roc_auc_score(y_test, pred_test[:,1])))
```

Output:

```
Train set
Logistic regression roc-auc: 0.7672664367367301
Test set
Logistic regression roc-auc: 0.7957738095238095
```

## Gradient Boosted Classifier

```python
gbc = GradientBoostingClassifier(n_estimators=300, random_state=44)
gbc.fit(X_train[['Cabin_mapped', 'Sex']], y_train)
pred_train = gbc.predict_proba(X_train[['Cabin_mapped', 'Sex']])
pred_test = gbc.predict_proba(X_test[['Cabin_mapped', 'Sex']].fillna(0))

print('Train set')
print('Gradient Boosted Trees roc-auc: {}'.format(roc_auc_score(y_train, pred_train[:,1])))
print('Test set')
print('Gradient Boosted Trees roc-auc: {}'.format(roc_auc_score(y_test, pred_test[:,1])))
```

Output:

```
Train set
Gradient Boosted Trees roc-auc: 0.8731860480249887
Test set
Gradient Boosted Trees roc-auc: 0.816845238095238
```

```python
gbc = GradientBoostingClassifier(n_estimators=300, random_state=44)
gbc.fit(X_train[['Cabin_reduced', 'Sex']], y_train)
pred_train = gbc.predict_proba(X_train[['Cabin_reduced', 'Sex']])
pred_test = gbc.predict_proba(X_test[['Cabin_reduced', 'Sex']].fillna(0))

print('Train set')
print('Gradient Boosted Trees roc-auc: {}'.format(roc_auc_score(y_train, pred_train[:,1])))
print('Test set')
print('Gradient Boosted Trees roc-auc: {}'.format(roc_auc_score(y_test, pred_test[:,1])))
```

Output:

```
Train set
Gradient Boosted Trees roc-auc: 0.8204756946703976
Test set
Gradient Boosted Trees roc-auc: 0.8332142857142857
```

## Conclusion

We can see that all the algorithms give better performance when the cardinality of the variables is low.