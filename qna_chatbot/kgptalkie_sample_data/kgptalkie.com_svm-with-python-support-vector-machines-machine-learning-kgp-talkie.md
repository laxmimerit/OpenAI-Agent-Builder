# [https://kgptalkie.com/svm-with-python-support-vector-machines-machine-learning-kgp-talkie](https://kgptalkie.com/svm-with-python-support-vector-machines-machine-learning-kgp-talkie)  
## SVM with Python | Support Vector Machines (SVM) | Machine Learning | KGP Talkie  

### Published by  
**KGP Talkie**  
**on 7 August 2020**  

---

## What is Support Vector Machines (SVM)  

We will start our discussion with a little introduction about **SVM**.  

**Support Vector Machine (SVM)** is a supervised **binary** classification algorithm. Given a set of points of two types in **N-dimensional** space, SVM generates a **(N−1) dimensional** hyperplane to separate those points into two groups.  

A **SVM** classifier would attempt to draw a **straight line** separating the **two sets** of data, and thereby create a **model** for **classification**. For **two dimensional** data like that shown here, this is a task we could do by hand. But immediately we see a problem: there is **more than one** possible dividing line that can perfectly **discriminate** between the two classes.  

---

## Support Vectors  

**Support vectors** are the data points, which are closest to the **hyperplane**. These points will define the **separating line** better by calculating **margins**. These points are more relevant to the **construction** of the **classifier**.  

---

## Hyperplane  

A **hyperplane** is a decision plane which **separates** between a set of objects having **different class** memberships.  

---

## Margin  

A **margin** is a gap between the two lines on the closest **class points**. This is calculated as the **perpendicular distance** from the line to **support vectors** or closest points. If the **margin** is larger in between the **classes**, then it is considered a **good margin**, a smaller margin is a **bad margin**.  

---

## How SVM Works?  

1. Generate **hyperplanes** which segregates the classes in the best way.  
   - Left-hand side figure showing **three hyperplanes** (black, blue, and orange).  
   - Here, the blue and orange have higher **classification error**, but the black is separating the two classes correctly.  

2. Select the right **hyperplane** with the maximum **segregation** from the either nearest data points as shown in the right-hand side figure.  

---

## Separation Planes  

### Linear  
### Non-Linear  

---

## Dealing with Non-Linear and Inseparable Planes  

**SVM** uses a **kernel** trick to transform the input space to a **higher dimensional** space.  

### Beauty of Kernel  

- **Kernels** allow us to do stuff in **infinite dimensions**.  
- Sometimes going to **higher dimension** is not just computationally **expensive**, but also **impossible**.  
- A function can be a **mapping** from **n-dimension** to **infinite dimension** which we may have little idea of how to deal with.  
- Then kernel gives us a wonderful shortcut.  

---

## SVM Kernels  

### Linear  
- A linear kernel can be used as a normal **dot product** between any two given observations.  
- The product between **two vectors** is the sum of the **multiplication** of each pair of **input values**.  
- **Training a SVM with a Linear Kernel is Faster** than with any other Kernel.  

### Polynomial  
- A **polynomial kernel** is a more generalized form of the **linear kernel**.  
- The polynomial kernel can distinguish curved or nonlinear **input space**.  

### Radial Basis Function (RBF)  
- The **RBF** kernel is a popular **kernel function** commonly used in **Support Vector Machine** classification.  
- **RBF** can map an input space in **infinite dimensional** space.  

---

## Let’s Build Model in `sklearn`  

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

cancer = datasets.load_breast_cancer()
cancer.keys()
```

**Output:**  
```
dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
```

```python
print(cancer.DESCR)
```

---

## Breast Cancer Wisconsin (Diagnostic) Dataset  

### Data Set Characteristics:  

- **Number of Instances:** 569  
- **Number of Attributes:** 30 numeric, predictive attributes and the class  
- **Class:**  
  - **WDBC-Malignant**  
  - **WDBC-Benign**  

### Summary Statistics:  

| Attribute | Min | Max |
|----------|-----|-----|
| radius (mean) | 6.981 | 28.11 |
| texture (mean) | 9.71 | 39.28 |
| ... | ... | ... |

**Source:** [https://kgptalkie.com/svm-with-python-support-vector-machines-machine-learning-kgp-talkie](https://kgptalkie.com/svm-with-python-support-vector-machines-machine-learning-kgp-talkie)  

---

## Code Implementation  

```python
X = cancer.data
y = cancer.target
X.shape, y.shape
```

**Output:**  
```
((569, 30), (569,))
```

```python
X[:2]
```

**Output:**  
```
array([[1.799e+01, 1.038e+01, 1.228e+02, 1.001e+03, 1.184e-01, 2.776e-01,
        3.001e-01, 1.471e-01, 2.419e-01, 7.871e-02, 1.095e+00, 9.053e-01,
        8.589e+00, 1.534e+02, 6.399e-03, 4.904e-02, 5.373e-02, 1.587e-02,
        3.003e-02, 6.193e-03, 2.538e+01, 1.733e+01, 1.846e+02, 2.019e+03,
        1.622e-01, 6.656e-01, 7.119e-01, 2.654e-01, 4.601e-01, 1.189e-01],
       [2.057e+01, 1.777e+01, 1.329e+02, 1.326e+03, 8.474e-02, 7.864e-02,
        8.690e-02, 7.017e-02, 1.812e-01, 5.667e-02, 5.435e-01, 7.339e-01,
        3.398e+00, 7.408e+01, 5.225e-03, 1.308e-02, 1.860e-02, 1.340e-02,
        1.389e-02, 3.532e-03, 2.499e+01, 2.341e+01, 1.588e+02, 1.956e+03,
        1.238e-01, 1.866e-01, 2.416e-01, 1.860e-01, 2.750e-01, 8.902e-02]])
```

```python
y[:10]
```

**Output:**  
```
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
```

---

## Standardization  

**Standardization** of a dataset is a common requirement for many **machine learning** estimators: they might behave badly if the individual feature do not more or less look like standard **normally distributed** data (e.g. Gaussian with **0** mean and **unit** variance).  

The idea behind **StandardScaler()** is that it will transform your data such that its **distribution** will have a **mean** value **0** and **standard deviation** of **1**.  

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## Split the Data and Build the Model  

```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1, stratify=y)
```

---

## Linear Kernel  

```python
from sklearn import svm
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)

print('Accuracy: ', metrics.accuracy_score(y_test, y_predict))
print('Precision: ', metrics.precision_score(y_test, y_predict))
print('Recall: ', metrics.recall_score(y_test, y_predict))
```

**Output:**  
```
Accuracy:  0.9649122807017544
Precision:  0.9594594594594594
Recall:  0.9861111111111112
```

```python
print('Confusion Matrix')
mat = metrics.confusion_matrix(y_test, y_predict)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
           xticklabels=cancer.target_names,
           yticklabels=cancer.target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
```

---

## Polynomial Kernel  

```python
clf = svm.SVC(kernel='poly', degree=5, gamma=100)
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)

print('Accuracy: ', metrics.accuracy_score(y_test, y_predict))
print('Precision: ', metrics.precision_score(y_test, y_predict))
print('Recall: ', metrics.recall_score(y_test, y_predict))
```

**Output:**  
```
Accuracy:  0.631578947368421
Precision:  0.631578947368421
Recall:  1.0
```

---

## Sigmoid Kernel  

```python
clf = svm.SVC(kernel='sigmoid', gamma=200, C=10000)
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)

print('Accuracy: ', metrics.accuracy_score(y_test, y_predict))
print('Precision: ', metrics.precision_score(y_test, y_predict))
print('Recall: ', metrics.recall_score(y_test, y_predict))
```

**Output:**  
```
Accuracy:  0.631578947368421
Precision:  0.631578947368421
Recall:  1.0
```

---

## `np.unique()`  

This function returns an **array of unique elements** in the input array. The function can be able to return a **tuple of array of unique values** and an array of associated indices. Nature of the indices depend upon the type of **return** parameter in the function call.  

```python
element, count = np.unique(y_test, return_counts=True)
element, count
```

**Output:**  
```
(array([0, 1]), array([42, 72], dtype=int64))
```

---

**Source:** [https://kgptalkie.com/svm-with-python-support-vector-machines-machine-learning-kgp-talkie](https://kgptalkie.com/svm-with-python-support-vector-machines-machine-learning-kgp-talkie)