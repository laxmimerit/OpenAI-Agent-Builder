https://kgptalkie.com/decision-tree-machine-learning-in-python-kgp-talkie

# Decision Tree Machine Learning in Python KGP Talkie

Published by  
KGP Talkie  
on  
7 August 2020  
7 August 2020  

For detailed theory, read **An introduction to Statistical Learning**:  
http://faculty.marshall.usc.edu/gareth-james/ISL/

## What is a Decision Tree?

A **decision tree** is a flowchart-like tree structure where:

- An **internal node** represents a feature  
- A **branch** represents a **decision rule**  
- Each **leaf node** represents the **outcome**  

The **topmost node** in a decision tree is known as the **root node**. It learns to partition on the basis of the **attribute value** recursively, a process called **recursive partitioning**. This structure helps in **decision making** and mimics **human-level thinking**, making it easy to understand and interpret.

---

## Why Decision Trees?

- **Mimic human-level thinking**: Easy to understand and interpret data  
- **Transparent logic**: Unlike black-box algorithms (e.g., SVM, NN), decision trees show the **logic** behind decisions  

---

## How Decision Trees Work

1. **Select the best attribute** using **Attribute Selection Measures (ASM)** to split the records  
2. **Make that attribute** a decision node and break the dataset into smaller subsets  
3. **Recursively repeat** the process for each child until one of these conditions is met:
   - All tuples belong to the same attribute value  
   - No more attributes remain  
   - No more instances  

### Algorithms for Building Decision Trees

- **CART (Classification and Regression Trees)**: Uses **Gini Index** (for classification)  
- **ID3 (Iterative Dichotomiser 3)**: Uses **Entropy** and **Information Gain**  

---

## Decision Making with Attribute Selection Measures (ASM)

### Information Gain

- **Entropy** measures the **uncertainty** in a dataset:  
  $$
  H(S) = \sum_{c=C} -p(c) \cdot \log_2 p(c)
  $$
  - $ S $: Current dataset  
  - $ C $: Set of classes in $ S $  
  - $ p(c) $: Probability of class $ c $  

- **Information Gain** calculates the **reduction in entropy** when splitting on an attribute:  
  $$
  IG(A,S) = H(S) - \sum_{t} p(t) \cdot H(t)
  $$
  - $ H(S) $: Entropy of set $ S $  
  - $ T $: Subsets created by splitting $ S $ on attribute $ A $  
  - $ H(t) $: Entropy of subset $ t $  

**Steps**:
1. Calculate entropy for all categorical values  
2. Take average information entropy for the current attribute  
3. Calculate gain for the current attribute  
4. Pick the attribute with the **highest gain**  

Repeat until the desired tree is built.

---

### Gain Ratio

- Corrects **information gain's bias** towards attributes with many values by adding **split information** in the denominator:  
  $$
  \text{SplitInformation}(S, A) = - \sum_{i=1}^{c} \frac{|S_i|}{|S|} \cdot \log_2 \frac{|S_i|}{|S|}
  $$
  $$
  \text{Gain Ratio}(S, A) \equiv \frac{\text{Gain}(S, A)}{\text{SplitInformation}(S, A)}
  $$

---

### Gini Index

- Measures the **likelihood of incorrect classification** for a new instance:  
  - **Pure dataset**: Gini Index = 0  
  - **Mixed dataset**: Gini Index = high  

---

## Optimizing Decision Trees

### Parameters

- **criterion**: `"gini"` (default) or `"entropy"`  
- **splitter**: `"best"` (default) or `"random"`  
- **max_depth**: Controls **overfitting** (higher = overfitting, lower = underfitting)  

---

## Recursive Binary Splitting

- All features are considered, and **split points** are tested using a **cost function**  
- The **best split** (lowest cost) is selected  

---

## When to Stop Splitting?

- **Pruning** removes branches with **low importance** to reduce complexity and **overfitting**  

---

## Decision Tree Regressor

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

plt.figure(figsize=(16, 4))
plt.plot(y_pred, label='y_pred')
plt.plot(y_test, label='y_test')
plt.xlabel('X_test', fontsize=14)
plt.ylabel('Value of y(pred , test)', fontsize=14)
plt.title('Comparing predicted values and true values')
plt.legend(title='Parameter where:')
plt.show()

np.sqrt(metrics.mean_squared_error(y_test, y_pred))
```

**Output**:
```
70.61829663921893
```

---

## Decision Tree as a Classifier

```python
from sklearn.tree import DecisionTreeClassifier
iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2, stratify=y)
clf = DecisionTreeClassifier(criterion='gini', random_state=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
```

**Output**:
```
Accuracy:  0.9666666666666667
```

### Confusion Matrix

```python
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

cm = confusion_matrix(y_test, y_pred)
fig, ax = plot_confusion_matrix(conf_mat=cm)
plt.title('Relative ratios between actual class and predicted class')
plt.show()
```

### Classification Report

```python
print(metrics.classification_report(y_test, y_pred))
```

**Output**:
```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           1       0.91      1.00      0.95        10
           2       1.00      0.90      0.95        10

    accuracy                           0.97        30
   macro avg       0.97      0.97      0.97        30
weighted avg       0.97      0.97      0.97        30
```

---

## References

- **Source**: https://kgptalkie.com/decision-tree-machine-learning-in-python-kgp-talkie  
- **ISL Book**: http://faculty.marshall.usc.edu/gareth-james/ISL/