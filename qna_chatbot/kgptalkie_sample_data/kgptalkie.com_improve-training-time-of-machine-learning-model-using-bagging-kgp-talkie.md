https://kgptalkie.com/improve-training-time-of-machine-learning-model-using-bagging-kgp-talkie

# Improve Training Time of Machine Learning Model Using Bagging | KGP Talkie

Published by  
KGP Talkie  
on  
7 August 2020  
7 August 2020  

## How Bagging Works

First of all, we will try to understand what **Bagging** is from the following diagram:

Let’s say we have a dataset to train the model. First, we need to divide this dataset into **number of datasets** (at least more than 2). Then, we need to apply the `classifier` on each of the datasets separately. Finally, we do **aggregation** to get the **output**.

### Time Complexity Example

SVM time complexity = $ O(n^3) $, i.e., as we increase the number of input samples, training time increases cubically.

**Example:**  
If 1000 input samples take 10 seconds to train, then 3000 input samples might take $ 10 \times 3^3 $ seconds to train.

If we divide 3000 samples into 3 categories (each dataset contains 1000 samples), training each dataset will take 10 seconds. The overall training time will be **30 seconds** (10 + 10 + 10). This improves the **training time** of the **machine learning** model.

Instead of 270 seconds, dividing into 3 sets reduces the time to 30 seconds.

---

## Implementation Example

### Importing Required Libraries

```python
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn import datasets
from sklearn.svm import SVC
```

### Importing Iris Dataset

```python
iris = datasets.load_iris()
iris.keys()
```

Output:
```
dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
```

### Description of the Iris Dataset

```python
print(iris.DESCR)
```

**Source:** https://kgptalkie.com/improve-training-time-of-machine-learning-model-using-bagging-kgp-talkie

**Data Set Characteristics:**

- **Number of Instances:** 150 (50 in each of three classes)
- **Number of Attributes:** 4 numeric, predictive attributes and the class
- **Attribute Information:**
  - sepal length in cm
  - sepal width in cm
  - petal length in cm
  - petal width in cm
  - class:
    - Iris-Setosa
    - Iris-Versicolour
    - Iris-Virginica

**Summary Statistics:**

**Source:** https://kgptalkie.com/improve-training-time-of-machine-learning-model-using-bagging-kgp-talkie

| Attribute        | Min | Max | Mean | SD   | Class Correlation |
|------------------|-----|-----|------|------|-------------------|
| sepal length     | 4.3 | 7.9 | 5.84 | 0.83 | 0.7826            |
| sepal width      | 2.0 | 4.4 | 3.05 | 0.43 | -0.4194           |
| petal length     | 1.0 | 6.9 | 3.76 | 1.76 | 0.9490 (high!)    |
| petal width      | 0.1 | 2.5 | 1.20 | 0.76 | 0.9565 (high!)    |

**Source:** https://kgptalkie.com/improve-training-time-of-machine-learning-model-using-bagging-kgp-talkie

The famous Iris database, first used by Sir R.A. Fisher. The dataset is taken from Fisher's paper. Note that it's the same as in R, but not as in the UCI Machine Learning Repository, which has two wrong data points.

This is perhaps the best known database to be found in the pattern recognition literature. Fisher's paper is a classic in the field and is referenced frequently to this day. (See Duda & Hart, for example.) The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other.

### Data Preparation

```python
X = iris.data
y = iris.target
```

Let’s check the shape of these variables:

```python
X.shape, y.shape
```

Output:
```
((150, 4), (150,))
```

So, `X` has **150 samples** and **4 attributes**, while `y` has **150 samples**.

To increase the number of samples, we will use the `repeat()` function:

```python
X = np.repeat(X, repeats=500, axis=0)
y = np.repeat(y, repeats=500, axis=0)
```

New shape:

```python
X.shape, y.shape
```

Output:
```
((75000, 4), (75000,))
```

---

## Training Without Bagging

To train the model without bagging, we will create a classifier called `SVC()` with a **linear** kernel.

```python
%%time
clf = SVC(kernel='linear', probability=True, class_weight='balanced')
clf.fit(X, y)
print('SVC: ', clf.score(X, y))
```

Output:
```
SVC:  0.98
Wall time: 34.5 sec
```

---

## Training With Bagging

To train the model with bagging, we will create a classifier called `BaggingClassifier()` with a **linear** kernel.

```python
%%time
n_estimators = 10
clf = BaggingClassifier(SVC(kernel='linear', probability=True, class_weight='balanced'), n_estimators=n_estimators, max_samples=1.0/n_estimators)
clf.fit(X, y)
print('SVC: ', clf.score(X, y))
```

Output:
```
SVC:  0.98
Wall time: 10.5 s
```

---

## Conclusion

From the above results, we can observe an improvement in training time of the model from **34.5 seconds** to **10.5 seconds** using **Bagging**.