# Feature Selection using Fisher Score and Chi2 (χ²) Test | Titanic Dataset | Machine Learning | KGP Talkie  
**Source:** https://kgptalkie.com/feature-selection-using-fisher-score-and-chi2-%CF%872-test-titanic-dataset-machine-learning-kgp-talkie  

---

## Published by  
KGP Talkie  
**Date:** 11 August 2020  

---

## Feature Selection using Fisher Score and Chi2 (χ²) Test  
**Watch Full Playlist:**  
https://www.youtube.com/playlist?list=PLc2rvfiptPSQYzmDIFuq2PqN2n28ZjxDH  

---

## What is Fisher Score and Chi2 (χ²) Test  

### Fisher Score  
- Fisher score is one of the most widely used supervised feature selection methods.  
- It selects each feature independently according to their **scores** under the Fisher criterion.  
- This leads to a **suboptimal subset** of features.  

### Chi Square (χ²) Test  
- A chi-squared test, also written as **X² test**, is any **statistical** hypothesis test where the sampling distribution of the test **statistic** is a chi-squared distribution.  
- The **chi-square test** measures dependence between **stochastic variables**.  
- Using this function **weeds out** the features that are the most likely to be independent of class and therefore irrelevant for classification.  

---

## Importing Required Libraries  

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.metrics import accuracy_score
```

---

## Loading the Required Dataset  

```python
titanic = sns.load_dataset('titanic')
titanic.head()
```

**Output:**  
| survived | pclass | sex   | age    | sibsp | parch | fare     | embarked | class  | who   | adult_male | deck | embark_town | alive | alone |
|----------|--------|-------|--------|-------|-------|----------|----------|--------|-------|------------|------|-------------|-------|-------|
| 0        | 3      | male  | 22.0   | 1     | 0     | 7.2500   | S        | Third  | man   | True       | NaN  | Southampton | no    | False |
| 1        | 1      | female| 38.0   | 1     | 0     | 71.2833  | C        | First  | woman | False      | C    | Cherbourg   | yes   | False |
| 1        | 3      | female| 26.0   | 0     | 0     | 7.9250   | S        | Third  | woman | False      | NaN  | Southampton | yes   | True  |
| 1        | 1      | female| 35.0   | 1     | 0     | 53.1000  | S        | First  | woman | False      | C    | Southampton | yes   | False |
| 0        | 3      | male  | 35.0   | 0     | 0     | 8.0500   | S        | Third  | man   | True       | NaN  | Southampton | no    | True  |

---

## Data Preprocessing  

```python
titanic.isnull().sum()
```

**Output:**  
```
survived         0
pclass           0
sex              0
age            177
sibsp            0
parch            0
fare             0
embarked         2
class            0
who              0
adult_male       0
deck           688
embark_town      2
alive            0
alone            0
dtype: int64
```

**Dropping columns and handling missing values:**  
```python
titanic.drop(labels=['age', 'deck'], axis=1, inplace=True)
titanic = titanic.dropna()
titanic.isnull().sum()
```

**Output:**  
```
survived       0
pclass         0
sex            0
sibsp          0
parch          0
fare           0
embarked       0
class          0
who            0
adult_male     0
embark_town    0
alive          0
alone          0
dtype: int64
```

**Creating a copy of the dataset:**  
```python
data = titanic[['pclass', 'sex', 'sibsp', 'parch', 'embarked', 'who', 'alone']].copy()
data.head()
```

**Encoding categorical variables:**  
```python
sex = {'male': 0, 'female': 1}
data['sex'] = data['sex'].map(sex)

ports = {'S': 0, 'C': 1, 'Q': 2}
data['embarked'] = data['embarked'].map(ports)

who = {'man': 0, 'woman': 1, 'child': 2}
data['who'] = data['who'].map(who)

alone = {True: 1, False: 0}
data['alone'] = data['alone'].map(alone)
```

---

## Do F_Score  

```python
X = data.copy()
y = titanic['survived']
X.shape, y.shape
```

**Output:**  
```
((889, 7), (889,))
```

**Train-test split:**  
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

**Chi2 test:**  
```python
f_score = chi2(X_train, y_train)
f_score
```

**Output:**  
```
(array([ 22.65169202, 152.91534343,   0.52934285,  10.35663782,
         16.13255653, 161.42431175,  13.4382363 ]),
 array([1.94189138e-06, 3.99737147e-35, 4.66883271e-01, 1.29009955e-03,
        5.90599986e-05, 5.52664700e-37, 2.46547298e-04]))
```

**Plotting p-values:**  
```python
p_values = pd.Series(f_score[1], index=X_train.columns)
p_values.sort_values(ascending=True, inplace=True)
p_values.plot.bar()
plt.title('pvalues with respect to features')
```

---

## Random Forest Classification  

```python
def run_randomForest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy: ', accuracy_score(y_test, y_pred))
```

**Testing with different feature subsets:**  
```python
X_train_2 = X_train[['who', 'sex']]
X_test_2 = X_test[['who', 'sex']]
%%time
run_randomForest(X_train_2, X_test_2, y_train, y_test)
```

**Output:**  
```
Accuracy:  0.7191011235955056
Wall time: 687 ms
```

```python
X_train_3 = X_train[['who', 'sex', 'pclass']]
X_test_3 = X_test[['who', 'sex', 'pclass']]
%%time
run_randomForest(X_train_3, X_test_3, y_train, y_test)
```

**Output:**  
```
Accuracy:  0.7415730337078652
Wall time: 649 ms
```

```python
X_train_4 = X_train[['who', 'sex', 'pclass', 'embarked']]
X_test_4 = X_test[['who', 'sex', 'pclass', 'embarked']]
%%time
run_randomForest(X_train_4, X_test_4, y_train, y_test)
```

**Output:**  
```
Accuracy:  0.7584269662921348
Wall time: 609 ms
```

```python
X_train_5 = X_train[['who', 'sex', 'pclass', 'embarked', 'alone']]
X_test_5 = X_test[['who', 'sex', 'pclass', 'embarked', 'alone']]
%%time
run_randomForest(X_train_5, X_test_5, y_train, y_test)
```

**Output:**  
```
Accuracy:  0.7528089887640449
Wall time: 413 ms
```

**Testing with the original dataset:**  
```python
%%time
run_randomForest(X_train, X_test, y_train, y_test)
```

**Output:**  
```
Accuracy:  0.7359550561797753
Wall time: 576 ms
```

---

**Source:** https://kgptalkie.com/feature-selection-using-fisher-score-and-chi2-%CF%872-test-titanic-dataset-machine-learning-kgp-talkie