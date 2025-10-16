# KNN with python | K Nearest Neighbors algorithm Machine Learning | KGP Talkie  
**Source:** https://kgptalkie.com/knn-with-python-k-nearest-neighbors-algorithm-machine-learning-kgp-talkie  

Published by  
**KGP Talkie**  
on  
**8 August 2020**  

---

## How does the KNN algorithm work?

In **KNN**, **K** is the number of nearest **neighbors**. The number of **neighbors** is the core deciding factor. **K** is generally an **odd** number if the number of classes is **2**. When **K=1**, then the algorithm is known as the **nearest neighbor algorithm**. This is the simplest case. Suppose **P1** is the point, for which label needs to **predict**. First, you find the one closest point to **P1** and then the label of the nearest point assigned to **P1**.

For finding closest similar points, you find the **distance** between points using distance measures such as Euclidean distance, Hamming distance, Manhattan distance, and Minkowski distance.

### KNN has the following basic steps:
1. **Calculate distance**
2. **Find closest neighbors**
3. **Vote for labels**

---

## Curse of dimensionality

To deal with the problem of the curse of dimensionality, you need to perform **principal component analysis** before applying any **machine learning** algorithm, or you can also use feature **selection approach**. Research has shown that in large dimension **Euclidean distance** is not useful anymore. Therefore, you can prefer other measures such as **cosine similarity**, which get decidedly less affected by high dimension.

---

## How do you decide the number of neighbors in KNN?

You can watch this video for better understanding.  
**Source:** https://kgptalkie.com/knn-with-python-k-nearest-neighbors-algorithm-machine-learning-kgp-talkie  

Now, you understand the **KNN** algorithm working mechanism. At this point, the question arises that how to choose the **optimal** number of neighbors? And what are its effects on the **classifier**? The number of neighbors (**K**) in **KNN** is a **hyperparameter** that you need to choose at the time of model building. You can think of **K** as a controlling variable for the prediction model.

Research has shown that no **optimal** number of neighbors suits all kinds of data sets. Each dataset has its own requirements. In the case of a small number of neighbors, the noise will have a higher influence on the result, and a large number of neighbors make it computationally **expensive**. Research has also shown that a small amount of neighbors are most flexible fit, which will have low **bias** but high **variance**, and a large number of neighbors will have a smoother decision boundary, which means lower **variance** but higher **bias**.

---

## Classifier Building in Python and Scikit-learn

You can use the **wine** dataset, which is a very famous multi-class classification problem. This data is the result of a **chemical analysis** of wines grown in the same region in Italy using three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines.

### Code Example:
```python
from sklearn import datasets
wine = datasets.load_wine()
wine.keys()
```

**Output:**
```
dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names'])
```

```python
print(wine.DESCR)
```

---

### Wine Recognition Dataset

**Source:** https://kgptalkie.com/knn-with-python-k-nearest-neighbors-algorithm-machine-learning-kgp-talkie  

#### Data Set Characteristics:

- **Number of Instances:** 178 (50 in each of three classes)
- **Number of Attributes:** 13 numeric, predictive attributes and the class
- **Attribute Information:**
  - Alcohol
  - Malic acid
  - Ash
  - Alcalinity of ash
  - Magnesium
  - Total phenols
  - Flavanoids
  - Nonflavanoid phenols
  - Proanthocyanins
  - Color intensity
  - Hue
  - OD280/OD315 of diluted wines
  - Proline

**Source:** https://kgptalkie.com/knn-with-python-k-nearest-neighbors-algorithm-machine-learning-kgp-talkie  

#### Class Distribution:
- **class_0**
- **class_1**
- **class_2**

#### Summary Statistics:

| Attribute                     | Min   | Max   | Mean  | SD    |
|-----------------------------|-------|-------|-------|-------|
| Alcohol                     | 11.0  | 14.8  | 13.0  | 0.8   |
| Malic Acid                  | 0.74  | 5.80  | 2.34  | 1.12  |
| Ash                         | 1.36  | 3.23  | 2.36  | 0.27  |
| Alcalinity of Ash           | 10.6  | 30.0  | 19.5  | 3.3   |
| Magnesium                   | 70.0  | 162.0 | 99.7  | 14.3  |
| Total Phenols               | 0.98  | 3.88  | 2.29  | 0.63  |
| Flavanoids                  | 0.34  | 5.08  | 2.03  | 1.00  |
| Nonflavanoid Phenols        | 0.13  | 0.66  | 0.36  | 0.12  |
| Proanthocyanins             | 0.41  | 3.58  | 1.59  | 0.57  |
| Colour Intensity            | 1.3   | 13.0  | 5.1   | 2.3   |
| Hue                         | 0.48  | 1.71  | 0.96  | 0.23  |
| OD280/OD315 of diluted wines | 1.27 | 4.00  | 2.61  | 0.71  |
| Proline                     | 278   | 1680  | 746   | 315   |

**Source:** https://kgptalkie.com/knn-with-python-k-nearest-neighbors-algorithm-machine-learning-kgp-talkie  

**Missing Attribute Values:** None  
**Class Distribution:** class_0 (59), class_1 (71), class_2 (48)  
**Creator:** R.A. Fisher  
**Donor:** Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)  
**Date:** July, 1988  

This is a copy of UCI ML Wine recognition datasets.  
**Source:** https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data  

---

### Dataset Example:
```python
wine.data[:3]
```

**Output:**
```
array([[1.423e+01, 1.710e+00, 2.430e+00, 1.560e+01, 1.270e+02, 2.800e+00,
        3.060e+00, 2.800e-01, 2.290e+00, 5.640e+00, 1.040e+00, 3.920e+00,
        1.065e+03],
       [1.320e+01, 1.780e+00, 2.140e+00, 1.120e+01, 1.000e+02, 2.650e+00,
        2.760e+00, 2.600e-01, 1.280e+00, 4.380e+00, 1.050e+00, 3.400e+00,
        1.050e+03],
       [1.316e+01, 2.360e+00, 2.670e+00, 1.860e+01, 1.010e+02, 2.800e+00,
        3.240e+00, 3.000e-01, 2.810e+00, 5.680e+00, 1.030e+00, 3.170e+00,
        1.185e+03]])
```

---

## Import Libraries

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
```

```python
X = wine.data
y = wine.target
X.shape, y.shape
```

**Output:**
```
((178, 13), (178,))
```

---

## Splitting Data

To understand model **performance**, dividing the dataset into a training set and a test set is a good strategy.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
```

---

## Generating Model for K=3

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_predict = knn.predict(X_test)
print('Accuracy: ', metrics.accuracy_score(y_test, y_predict))
```

**Output:**
```
Accuracy:  0.6851851851851852
```

---

## Generating Model for K=5

```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_predict = knn.predict(X_test)
print('Accuracy: ', metrics.accuracy_score(y_test, y_predict))
```

**Output:**
```
Accuracy:  0.7222222222222222
```

---

## Generating Model for K=7

```python
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
y_predict = knn.predict(X_test)
print('Accuracy: ', metrics.accuracy_score(y_test, y_predict))
```

**Output:**
```
Accuracy:  0.7407407407407407
```

---

## Standardization

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

```python
X_scaled[:3]
```

**Output:**
```
array([[ 1.51861254, -0.5622498 ,  0.23205254, -1.16959318,  1.91390522,
         0.80899739,  1.03481896, -0.65956311,  1.22488398,  0.25171685,
         0.36217728,  1.84791957,  1.01300893],
       [ 0.24628963, -0.49941338, -0.82799632, -2.49084714,  0.01814502,
         0.56864766,  0.73362894, -0.82071924, -0.54472099, -0.29332133,
         0.40605066,  1.1134493 ,  0.96524152],
       [ 0.19687903,  0.02123125,  1.10933436, -0.2687382 ,  0.08835836,
         0.80899739,  1.21553297, -0.49840699,  2.13596773,  0.26901965,
         0.31830389,  0.78858745,  1.39514818]])
```

---

## Cross-Validation and Optimal K

```python
from sklearn.model_selection import cross_val_score
neighbors = list(range(1, 50, 2))
cv_scores = []
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_scaled, y, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
```

```python
cv_scores[:5]
```

**Output:**
```
[0.9434640522875817,
 0.9545751633986927,
 0.9604575163398692,
 0.9663398692810456,
 0.9718954248366012]
```

```python
MSE = [1 - x for x in cv_scores]
MSE[:5]
```

**Output:**
```
[0.05653594771241832,
 0.04542483660130725,
 0.0395424836601308,
 0.03366013071895435,
 0.028104575163398815]
```

```python
optimal_k = neighbors[MSE.index(min(MSE))]
print('The optimal number of k is: ', optimal_k)
```

**Output:**
```
The optimal number of k is:  23
```

---

## Final Model with Optimal K

```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0, stratify=y)
knn = KNeighborsClassifier(n_neighbors=23)
knn.fit(X_train, y_train)
y_predict = knn.predict(X_test)
print('Accuracy: ', metrics.accuracy_score(y_test, y_predict))
```

**Output:**
```
Accuracy:  0.9814814814814815
```

---

## Plotting Error vs K

```python
plt.plot(neighbors, MSE)
plt.xlabel('Number of K')
plt.ylabel('Error')
plt.title('Variation of error with changing K')
plt.show()
```

**Source:** https://kgptalkie.com/knn-with-python-k-nearest-neighbors-algorithm-machine-learning-kgp-talkie