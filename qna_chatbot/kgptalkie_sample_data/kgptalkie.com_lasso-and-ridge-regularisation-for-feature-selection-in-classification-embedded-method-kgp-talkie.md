https://kgptalkie.com/lasso-and-ridge-regularisation-for-feature-selection-in-classification-embedded-method-kgp-talkie

# Lasso and Ridge Regularisation for Feature Selection in Classification | Embedded Method | KGP Talkie

**Source:** [https://kgptalkie.com/lasso-and-ridge-regularisation-for-feature-selection-in-classification-embedded-method-kgp-talkie](https://kgptalkie.com/lasso-and-ridge-regularisation-for-feature-selection-in-classification-embedded-method-kgp-talkie)

## Published by
KGP Talkie  
**Date:** 8 August 2020

---

## What is Regularisation?

Regularization adds a penalty on the different parameters of the model to reduce the **freedom** of the model. Hence, the model will be less likely to fit the **noise** of the training data and will improve the **generalization** abilities of the model.

### Types of Regularization
There are basically 3 types of regularization:

1. **L1 regularization (also called Lasso)**  
   Shrinks the coefficients which are less important to **zero**. That means with **Lasso regularization**, we can remove some features.

2. **L2 regularization (also called Ridge)**  
   Doesn’t reduce the coefficients to zero but reduces the regression coefficients. With this reduction, we can identify which feature has more importance.

3. **L1/L2 regularization (also called Elastic net)**

---

## What is Lasso Regularisation?

### 3 Sources of Error
1. **Noise** – We can’t do anything with the noise. Let’s focus on the following errors.
2. **Bias error** – It quantifies how much on average the predicted values are different from the actual value.
3. **Variance** – Quantifies how predictions made on the same observation differ from each other.

### Bias-Variance Trade-off
By increasing **model complexity**, total error will decrease till some point and then start to increase. We need to select **optimum model complexity** to get less error.

- **Low complexity model**: high bias and low variance  
- **High complexity model**: low bias and high variance  

If you are getting **high bias**, increase **model complexity**. If you are getting **high variance**, decrease **model complexity**. That’s how any **machine learning** algorithm works.

---

### L1 Regularization Formula
The L1 regularization adds a **penalty** equal to the **sum** of the **absolute value** of the **coefficients**.

$$
\text{Penalty} = \lambda \sum_{i=1}^{n} |w_i|
$$

Where:
- $ w $ is the regression coefficient
- $ \lambda $ is the regularization coefficient

**Observation:**  
L1 regularization will shrink some parameters to **zero**, hence some variables will not play any role in the model to get the **final output**. L1 regression can be seen as a way to select features in a model.

---

### Choosing $ \lambda $
To choose the best $ \lambda $, we can split the data into 3 sets:

1. **Training set** – Fit the model and set regression coefficients with regularization.
2. **Validation set** – Test the model’s performance to select $ \lambda $.
3. **Test set** – Generalize testing on the test set.

---

## What is Ridge Regularisation?

### L2 Regularization Formula
The L2 regularization adds a penalty equal to the **sum of the squared value** of the coefficients.

$$
\text{Penalty} = \lambda \sum_{i=1}^{n} w_i^2
$$

Where:
- $ \lambda $ is the tuning parameter or optimization parameter
- $ w $ is the regression coefficient

**Observation:**
- If $ \lambda $ is high → high bias and low variance
- If $ \lambda $ is low → low bias and high variance

The **L2 regularization** will force the parameters to be relatively **small**. The bigger the penalization, the smaller (and the more robust) the coefficients are.

When compared to **L1 regularization**, the **coefficients** decrease progressively and are not cut to zero. They slowly decrease to **zero**.

---

## Practical Implementation with Titanic Dataset

### Importing Required Libraries
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
```

### Load and Preprocess Data
```python
titanic = sns.load_dataset('titanic')
titanic.isnull().sum()
titanic.drop(labels=['age', 'deck'], axis=1, inplace=True)
titanic = titanic.dropna()
titanic.isnull().sum()
```

### Data Encoding
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

### Split Data
```python
X = data.copy()
y = titanic['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```

### Feature Selection with Lasso
```python
sel = SelectFromModel(LogisticRegression(C=0.05, penalty='l1', solver='liblinear'))
sel.fit(X_train, y_train)
features = X_train.columns[sel.get_support()]
X_train_l1 = sel.transform(X_train)
X_test_l1 = sel.transform(X_test)
```

### Model Evaluation
```python
def run_randomForest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy: ', accuracy_score(y_test, y_pred))

%%time
run_randomForest(X_train_l1, X_test_l1, y_train, y_test)
```

**Output:**
```
Accuracy:  0.826530612244898
Wall time: 517 ms
```

---

## Ridge Regression

### RidgeClassifier
```python
from sklearn.linear_model import RidgeClassifier
rr = RidgeClassifier(alpha=300)
rr.fit(X_train, y_train)
rr.score(X_test, y_test)
```

**Output:**
```
0.8231292517006803
```

### RidgeClassifierCV
```python
from sklearn.linear_model import RidgeClassifierCV
rr = RidgeClassifierCV(alphas=[10, 20, 50, 100, 200, 300], cv=10)
rr.fit(X_train, y_train)
rr.score(X_test, y_test)
```

**Output:**
```
0.8197278911564626
```

**Best Alpha:**
```python
rr.alpha_
```

**Output:**
```
200
```