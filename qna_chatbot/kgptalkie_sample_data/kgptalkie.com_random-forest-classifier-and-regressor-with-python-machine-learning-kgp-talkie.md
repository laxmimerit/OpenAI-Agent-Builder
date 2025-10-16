https://kgptalkie.com/random-forest-classifier-and-regressor-with-python-machine-learning-kgp-talkie

# Random Forest Classifier and Regressor with Python | Machine Learning | KGP Talkie

Published by  
**KGP Talkie**  
on  
**7 August 2020**  
**7 August 2020**

## What is it?

A **Random Forest** is an ensemble technique capable of performing both **regression** and **classification** tasks using **multiple decision trees** and a technique called **Bootstrap** and **Aggregation** (commonly known as **bagging**). The basic idea is to combine **multiple decision trees** to determine the final output rather than relying on individual decision trees.

**Random forests** have various applications, such as recommendation engines, image classification, and feature selection. It can classify loyal loan applicants, identify fraudulent activity, and **predict** diseases. It forms the basis of the **Boruta algorithm**, which selects important features in a dataset.

## How the Random Forest Algorithm Works

1. **Select** random samples from a given dataset.
2. **Construct** a decision tree for each sample and get a **prediction result** from each **decision tree**.
3. **Perform a vote** for each **predicted result**.
4. **Select** the **prediction result** with the most votes as the final prediction.

## Important Feature for Classification

**Random forest** uses **gini importance** or **mean decrease in impurity (MDI)** to calculate the importance of each feature.

- **Gini importance** is also known as the total decrease in **node impurity**. This indicates how much the model fit or **accuracy** decreases when a variable is dropped. The larger the decrease, the more significant the variable is.
- **Mean decrease** is a significant parameter for variable selection.
- The **Gini index** can describe the overall explanatory power of the variables.

## Random Forests vs Decision Trees

| Feature                  | Random Forests                          | Decision Trees                        |
|-------------------------|-----------------------------------------|---------------------------------------|
| Composition             | A set of **multiple decision trees**    | Single tree                         |
| Overfitting             | Prevents overfitting via random subsets | **Deep decision trees** may overfit |
| Computational Speed     | **Slower**                              | **Faster**                          |
| Interpretability        | **Difficult to interpret**              | **Easily interpretable**            |
| Conversion to Rules     | No                                      | Can be converted to **rules**       |

## Part 1: Random Forest as a Regression

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

diabetes = datasets.load_diabetes()
diabetes.keys()
```

**Output:**
```
dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])
```

```python
print(diabetes.DESCR)
```

### Diabetes Dataset

**Data Set Characteristics:**

- **Source:** https://kgptalkie.com/random-forest-classifier-and-regressor-with-python-machine-learning-kgp-talkie
- **Number of Instances:** 442
- **Number of Attributes:** First 10 columns are numeric predictive values
- **Target:** Column 11 is a quantitative measure of disease progression one year after baseline
- **Attribute Information:**
  - age: age in years
  - sex
  - bmi: body mass index
  - bp: average blood pressure
  - s1: tc, T-Cells (a type of white blood cells)
  - s2: ldl, low-density lipoproteins
  - s3: hdl, high-density lipoproteins
  - s4: tch, thyroid stimulating hormone
  - s5: ltg, lamotrigine
  - s6: glu, blood sugar level

**Source:** https://kgptalkie.com/random-forest-classifier-and-regressor-with-python-machine-learning-kgp-talkie

**Note:** Each of these 10 feature variables have been mean centered and scaled by the standard deviation times `n_samples` (i.e., the sum of squares of each column totals 1).

```python
X = diabetes.data
y = diabetes.target
X.shape, y.shape
```

**Output:**
```
((442, 10), (442,))
```

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
```

### Plotting Predicted vs. Actual Values

```python
plt.figure(figsize=(16, 4))
plt.plot(y_pred, label='y_pred')
plt.plot(y_test, label='y_test')
plt.xlabel('X_test', fontsize=14)
plt.ylabel('Value of y(pred , test)', fontsize=14)
plt.title('Comparing predicted values and true values')
plt.legend(title='Parameter where:')
plt.show()
```

### Root Mean Square Error

```python
np.sqrt(metrics.mean_squared_error(y_test, y_pred))
```

**Output:**
```
53.505825893179875
```

```python
(72.78 - 53.50) / 72.78
```

**Output:**
```
0.26490794174223686
```

```python
y_test.std()
```

**Output:**
```
73.47317715932746
```

## Random Forest as a Classifier with Iris Dataset

```python
from sklearn.ensemble import RandomForestClassifier
iris = datasets.load_iris()
iris.target_names
```

**Output:**
```
array(['setosa', 'versicolor', 'virginica'], dtype='<U10')
```

```python
print(iris.DESCR)
```

### Iris Plants Dataset

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

| Attribute              | Min | Max | Mean | SD   | Class Correlation |
|-----------------------|-----|-----|------|------|-------------------|
| sepal length:         | 4.3 | 7.9 | 5.84 | 0.83 | 0.7826            |
| sepal width:          | 2.0 | 4.4 | 3.05 | 0.43 | -0.4194           |
| petal length:         | 1.0 | 6.9 | 3.76 | 1.76 | 0.9490 (high!)    |
| petal width:          | 0.1 | 2.5 | 1.20 | 0.76 | 0.9565 (high!)    |

**Source:** https://kgptalkie.com/random-forest-classifier-and-regressor-with-python-machine-learning-kgp-talkie

**Note:** The dataset is taken from Fisher's paper. It is the same as in R, but not as in the UCI Machine Learning Repository, which has two wrong data points.

### Using Iris Dataset

```python
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3, stratify=y)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
```

### Predicting Labels

```python
y_pred = clf.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
```

**Output:**
```
0.9777777777777777
```

```python
mat = metrics.confusion_matrix(y_test, y_pred)
mat
```

**Output:**
```
array([[15,  0,  0],
       [ 0, 15,  0],
       [ 0,  1, 14]], dtype=int64)
```

### Confusion Matrix

- **Diagonal elements** represent the number of points where the **predicted label** equals the **true label**.
- **Off-diagonal elements** are mislabeled by the classifier.
- Higher diagonal values indicate better performance.

### Plotting Confusion Matrix

```python
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plot_confusion_matrix(conf_mat=cm)
plt.title('Relative ratios between actual class and predicted class')
plt.show()
```

### Feature Importances

```python
clf.feature_importances_
```

**Output:**
```
array([0.1160593 , 0.03098375, 0.43034957, 0.42260737])
```

```python
iris.feature_names
```

**Output:**
```
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
```