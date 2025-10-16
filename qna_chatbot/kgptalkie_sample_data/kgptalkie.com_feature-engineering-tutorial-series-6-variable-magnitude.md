# Feature Engineering Tutorial Series 6: Variable magnitude  
**Source:** https://kgptalkie.com/feature-engineering-tutorial-series-6-variable-magnitude  

---

## Published Information  
**Author:** georgiannacambel  
**Date:** 4 October 2020  

---

## Does the magnitude of the variable matter?

In Linear Regression models, the scale of variables used to estimate the output matters. Linear models are of the type:

```python
y = w * x + b
```

where the regression coefficient `w` represents the expected change in `y` for a one unit change in `x` (the predictor). Thus, the magnitude of `w` is partly determined by the magnitude of the units being used for `x`. If `x` is a distance variable, just changing the scale from kilometers to miles will cause a change in the magnitude of the coefficient.

In addition, in situations where we estimate the outcome `y` by contemplating multiple predictors `x1, x2, ..., xn`, predictors with greater numeric ranges dominate over those with smaller numeric ranges.

Gradient descent converges faster when all the predictors (`x1` to `xn`) are within a similar scale, therefore having features in a similar scale is useful for Neural Networks as well as.

In Support Vector Machines, feature scaling can decrease the time required to find the support vectors.

Finally, methods using Euclidean distances or distances in general are also affected by the magnitude of the features, as Euclidean distance is sensitive to variations in the magnitude or scales of the predictors. Therefore feature scaling is required for methods that utilise distance calculations like k-nearest neighbours (KNN) and k-means clustering.

---

## In short:

**Magnitude matters because:**

- The regression coefficient is directly influenced by the scale of the variable  
- Variables with bigger magnitude / value range dominate over the ones with smaller magnitude / value range  
- Gradient descent converges faster when features are on similar scales  
- Feature scaling helps decrease the time to find support vectors for SVMs  
- Euclidean distances are sensitive to feature magnitude.

---

## The machine learning models affected by the magnitude of the feature are:

- Linear and Logistic Regression  
- Neural Networks  
- Support Vector Machines  
- KNN  
- K-means clustering  
- Linear Discriminant Analysis (LDA)  
- Principal Component Analysis (PCA)  

**Machine learning models insensitive to feature magnitude are the ones based on Trees:**

- Classification and Regression Trees  
- Random Forests  
- Gradient Boosted Trees  

---

## In this Blog

We will study the effect of feature magnitude on the performance of different **machine learning** algorithms.  
We will use the **Titanic dataset**.

---

## Let’s Start!

We will start by importing the necessary libraries.

```python
# to read the dataset into a dataframe and perform operations on it
import pandas as pd

# to perform basic array operations
import numpy as np

# import several machine learning algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# to scale the features
from sklearn.preprocessing import MinMaxScaler

# to evaluate performance and separate into train and test set
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
```

---

## Load data with numerical variables only

We will start by loading only the variables having numeric values from the **titanic** dataset.

```python
data = pd.read_csv('https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/titanic.csv',
                   usecols=['Pclass', 'Age', 'Fare', 'Survived'])
data.head()
```

**Output:**

```
   Survived  Pclass     Age     Fare
0         0       3  22.0   7.2500
1         1       1  38.0  71.2833
2         1       3  26.0   7.9250
3         1       1  35.0  53.1000
4         0       3  35.0   8.0500
```

Now we will have a look at the values of those variables to get an idea of the feature magnitudes.

```python
data.describe()
```

**Output:**

```
       Survived      Pclass         Age        Fare
count  891.000000  891.000000  714.000000  891.000000
mean     0.383838    2.308642   29.699118   32.204208
std      0.486592    0.836071   14.526497   49.693429
min      0.000000    1.000000    0.420000    0.000000
25%      0.000000    2.000000   20.125000    7.910400
50%      0.000000    3.000000   28.000000   14.454200
75%      1.000000    3.000000   38.000000   31.000000
max      1.000000    3.000000   80.000000  512.329200
```

We can see that **Fare** varies between 0 and 512, **Age** between 0 and 80, and **Pclass** between 1 and 3. So the variables have different magnitudes.

Let’s calculate the range of each variable. The range of a set of data is the difference between the largest and smallest values.

```python
for col in ['Pclass', 'Age', 'Fare']:
    print(col, 'range: ', data[col].max() - data[col].min())
```

**Output:**

```
Pclass range:  2
Age range:  79.58
Fare range:  512.3292
```

The range of values that each variable takes are quite different.

---

## Splitting the data

Now we will split the data into training and testing set with the help of `train_test_split()`. We will use the variables `Pclass`, `Age` and `Fare` as the feature space and `Survived` as the target. The `test_size = 0.3` will keep 30% data for testing and 70% data will be used for training the model. `random_state` controls the shuffling applied to the data before applying the split. The **titanic** dataset contains missing information so for this demo, we will fill those with 0s using `fillna()`.

```python
X_train, X_test, y_train, y_test = train_test_split(
    data[['Pclass', 'Age', 'Fare']].fillna(0),
    data.Survived,
    test_size=0.3,
    random_state=0)
```

**Output:**

```
X_train.shape, X_test.shape
((623, 3), (268, 3))
```

The training dataset contains 623 rows while the test dataset contains 268 rows.

---

## Feature Scaling

For this demonstration, we will scale the features between 0 and 1, using the `MinMaxScaler` from scikit-learn. To learn more about this scaling visit the [Scikit-Learn website](https://scikit-learn.org/stable/).

The transformation is given by:

```python
X_rescaled = (X – X.min) / (X.max – X.min)
```

And to transform the re-scaled features back to their original magnitude:

```python
X = X_rescaled * (max – min) + min
```

We will first initialize `scalar`. Then we will fit the `scalar` to the training dataset. Using this `scalar` we will transform `X_train` as well as `X_test`.

```python
# call the scaler
scaler = MinMaxScaler()

# fit the scaler
scaler.fit(X_train)

# re scale the datasets
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Let’s have a look at the scaled training dataset.

```python
print('Mean: ', X_train_scaled.mean(axis=0))
print('Standard Deviation: ', X_train_scaled.std(axis=0))
print('Minimum value: ', X_train_scaled.min(axis=0))
print('Maximum value: ', X_train_scaled.max(axis=0))
```

**Output:**

```
Mean:  [0.64365971 0.30131421 0.06335433]
Standard Deviation:  [0.41999093 0.21983527 0.09411705]
Minimum value:  [0. 0. 0.]
Maximum value:  [1. 1. 1.]
```

Now, the maximum values for all the features is 1, and the minimum value is zero, as expected. So they are in a similar scale.

---

## Logistic Regression

Let’s evaluate the effect of feature scaling on a Logistic Regression. We will first build the model using unscaled variables and then the scaled variables.

```python
# model built on unscaled variables
logit = LogisticRegression(
    random_state=44,
    C=1000,
    solver='lbfgs'
)
logit.fit(X_train, y_train)

print('Train set')
pred = logit.predict_proba(X_train)
print('Logistic Regression roc-auc: {}'.format(
    roc_auc_score(y_train, pred[:, 1])))
print('Test set')
pred = logit.predict_proba(X_test)
print('Logistic Regression roc-auc: {}'.format(
    roc_auc_score(y_test, pred[:, 1])))
```

**Output:**

```
Train set
Logistic Regression roc-auc: 0.7134823539619531
Test set
Logistic Regression roc-auc: 0.7080952380952381
```

Let’s look at the coefficients.

```python
logit.coef_
```

**Output:**

```
array([[-0.92585764, -0.01822689,  0.00233577]])
```

# model built on scaled variables

```python
logit = LogisticRegression(
    random_state=44,
    C=1000,
    solver='lbfgs'
)
logit.fit(X_train_scaled, y_train)

print('Train set')
pred = logit.predict_proba(X_train_scaled)
print('Logistic Regression roc-auc: {}'.format(
    roc_auc_score(y_train, pred[:, 1])))
print('Test set')
pred = logit.predict_proba(X_test_scaled)
print('Logistic Regression roc-auc: {}'.format(
    roc_auc_score(y_test, pred[:, 1])))
```

**Output:**

```
Train set
Logistic Regression roc-auc: 0.7134931997136721
Test set
Logistic Regression roc-auc: 0.7080952380952381
```

Let’s look at the coefficients.

```python
logit.coef_
```

**Output:**

```
array([[-1.85170244, -1.45782986,  1.19540159]])
```

---

## Support Vector Machines

```python
# model build on unscaled variables
SVM_model = SVC(random_state=44, probability=True, gamma='auto')
SVM_model.fit(X_train, y_train)

print('Train set')
pred = SVM_model.predict_proba(X_train)
print('SVM roc-auc: {}'.format(roc_auc_score(y_train, pred[:, 1])))
print('Test set')
pred = SVM_model.predict_proba(X_test)
print('SVM roc-auc: {}'.format(roc_auc_score(y_test, pred[:, 1])))
```

**Output:**

```
Train set
SVM roc-auc: 0.9016995292943755
Test set
SVM roc-auc: 0.6768154761904762
```

# model built on scaled variables

```python
SVM_model = SVC(random_state=44, probability=True, gamma='auto')
SVM_model.fit(X_train_scaled, y_train)

print('Train set')
pred = SVM_model.predict_proba(X_train_scaled)
print('SVM roc-auc: {}'.format(roc_auc_score(y_train, pred[:, 1])))
print('Test set')
pred = SVM_model.predict_proba(X_test_scaled)
print('SVM roc-auc: {}'.format(roc_auc_score(y_test, pred[:, 1])))
```

**Output:**

```
Train set
SVM roc-auc: 0.7047081408212403
Test set
SVM roc-auc: 0.6988690476190476
```

---

## K-Nearest Neighbours

```python
# model built on unscaled features
KNN = KNeighborsClassifier(n_neighbors=5)
KNN.fit(X_train, y_train)

print('Train set')
pred = KNN.predict_proba(X_train)
print('KNN roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
print('Test set')
pred = KNN.predict_proba(X_test)
print('KNN roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
```

**Output:**

```
Train set
KNN roc-auc: 0.8131141849360215
Test set
KNN roc-auc: 0.6947901111664178
```

# model built on scaled

```python
KNN = KNeighborsClassifier(n_neighbors=5)
KNN.fit(X_train_scaled, y_train)

print('Train set')
pred = KNN.predict_proba(X_train_scaled)
print('KNN roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
print('Test set')
pred = KNN.predict_proba(X_test_scaled)
print('KNN roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
```

**Output:**

```
Train set
KNN roc-auc: 0.826928785995703
Test set
KNN roc-auc: 0.7232453957192633
```

---

## Random Forests

```python
# model built on unscaled features
rf = RandomForestClassifier(n_estimators=200, random_state=39)
rf.fit(X_train, y_train)

print('Train set')
pred = rf.predict_proba(X_train)
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred[:, 1])))
print('Test set')
pred = rf.predict_proba(X_test)
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred[:, 1])))
```

**Output:**

```
Train set
Random Forests roc-auc: 0.9916108110453136
Test set
Random Forests roc-auc: 0.7614285714285715
```

# model built in scaled features

```python
rf = RandomForestClassifier(n_estimators=200, random_state=39)
rf.fit(X_train_scaled, y_train)

print('Train set')
pred = rf.predict_proba(X_train_scaled)
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
print('Test set')
pred = rf.predict_proba(X_test_scaled)
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
```

**Output:**

```
Train set
Random Forests roc-auc: 0.9916541940521898
Test set
Random Forests roc-auc: 0.7610714285714285
```

---

## AdaBoost

```python
# train adaboost on non-scaled features
ada = AdaBoostClassifier(n_estimators=200, random_state=44)
ada.fit(X_train, y_train)

print('Train set')
pred = ada.predict_proba(X_train)
print('AdaBoost roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
print('Test set')
pred = ada.predict_proba(X_test)
print('AdaBoost roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
```

**Output:**

```
Train set
AdaBoost roc-auc: 0.8477364916162339
Test set
AdaBoost roc-auc: 0.7733630952380953
```

# train adaboost on scaled features

```python
ada = AdaBoostClassifier(n_estimators=200, random_state=44)
ada.fit(X_train_scaled, y_train)

print('Train set')
pred = ada.predict_proba(X_train_scaled)
print('AdaBoost roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
print('Test set')
pred = ada.predict_proba(X_test_scaled)
print('AdaBoost roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
```

**Output:**

```
Train set
AdaBoost roc-auc: 0.8477364916162339
Test set
AdaBoost roc-auc: 0.7733630952380953
```

---

## Conclusion

Machine learning is like making a mixed fruit juice. If we want to get the best-mixed juice, we need to mix all fruit not by their size but based on their right proportion. We just need to remember apple and strawberry are not the same unless we make them similar in some context to compare their attribute. Similarly, in many machine learning algorithms, to bring all features in the same standing, we need to do scaling so that one significant number doesn’t impact the model just because of their large magnitude. Feature scaling in machine learning is one of the most critical steps during the pre-processing of data before creating a machine learning model. Scaling can make a difference between a weak machine learning model and a better one.

**Source:** https://kgptalkie.com/feature-engineering-tutorial-series-6-variable-magnitude  
**Source:** https://kgptalkie.com/feature-engineering-tutorial-series-6-variable-magnitude