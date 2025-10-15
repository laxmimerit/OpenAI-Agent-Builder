# Feature Selection with Filtering Method | Constant, Quasi Constant and Duplicate Feature Removal  
**Source:** [https://kgptalkie.com/feature-selection-with-filtering-method-constant-quasi-constant-and-duplicate-feature-removal](https://kgptalkie.com/feature-selection-with-filtering-method-constant-quasi-constant-and-duplicate-feature-removal)  

---

## Published by  
**KGP Talkie**  
**on 10 August 2020**  

---

## Watch Full Playlist  
[https://www.youtube.com/playlist?list=PLc2rvfiptPSQYzmDIFuq2PqN2n28ZjxDH](https://www.youtube.com/playlist?list=PLc2rvfiptPSQYzmDIFuq2PqN2n28ZjxDH)  

---

### Unnecessary and Redundant Features  
Unnecessary and redundant features:  
- Slow down training time  
- Affect algorithm performance  

### Advantages of Feature Selection  
- Higher model explainability  
- Easier implementation  
- Enhanced generalization (reduces overfitting)  
- Removes data redundancy  
- Lower training time  
- Less prone to errors  

---

## What is Filter Method?  
Features selected using **filter methods** can be used as input to any machine learning models.  

### Univariate Filter Methods  
- **Fisher Score**  
- **Mutual Information Gain**  
- **Variance**  

### Multivariate Filter Methods  
- Removes **redundant** features by considering mutual relationships between features.  

---

## Part 1 of Filtering Method  
## Part 2 of Filtering Method  

---

## Univariate Filtering Methods in this Lesson  
- **Constant Removal**  
- **Quasi Constant Removal**  
- **Duplicate Feature Removal**  

---

## Download Data Files  
[https://github.com/laxmimerit/Data-Files-for-Feature-Selection](https://github.com/laxmimerit/Data-Files-for-Feature-Selection)  

---

## Constant Feature Removal  

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
```

### Load Dataset  
```python
data = pd.read_csv('santander.csv', nrows=20000)
X = data.drop('TARGET', axis=1)
y = data['TARGET']
```

### Split Dataset  
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
```

### Constant Features Removal  
```python
constant_filter = VarianceThreshold(threshold=0)
constant_filter.fit(X_train)
constant_filter.get_support().sum()  # 291 features remaining
```

### Constant Features List  
```python
constant_list = [not temp for temp in constant_filter.get_support()]
X.columns[constant_list]
```

### Transform Data  
```python
X_train_filter = constant_filter.transform(X_train)
X_test_filter = constant_filter.transform(X_test)
```

---

## Quasi Constant Feature Removal  
```python
quasi_constant_filter = VarianceThreshold(threshold=0.01)
quasi_constant_filter.fit(X_train_filter)
quasi_constant_filter.get_support().sum()  # 245 features remaining
```

### Transform Data  
```python
X_train_quasi_filter = quasi_constant_filter.transform(X_train_filter)
X_test_quasi_filter = quasi_constant_filter.transform(X_test_filter)
```

---

## Remove Duplicate Features  
```python
X_train_T = X_train_quasi_filter.T
X_test_T = X_test_quasi_filter.T
X_train_T = pd.DataFrame(X_train_T)
X_test_T = pd.DataFrame(X_test_T)
```

### Find Duplicates  
```python
duplicated_features = X_train_T.duplicated()
features_to_keep = [not index for index in duplicated_features]
X_train_unique = X_train_T[features_to_keep].T
X_test_unique = X_test_T[features_to_keep].T
```

---

## Build ML Model and Compare Performance  

### Define Function  
```python
def run_randomForest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy on test set: ')
    print(accuracy_score(y_test, y_pred))
```

### Run on Transformed Data  
```python
run_randomForest(X_train_unique, X_test_unique, y_train, y_test)
# Accuracy: 0.95875
```

### Run on Original Data  
```python
run_randomForest(X_train, X_test, y_train, y_test)
# Accuracy: 0.9585
```

---

## Feature Selection with Filtering Method - Correlated Feature Removal  

### Correlation Matrix  
```python
corrmat = X_train_unique.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corrmat)
```

### Function to Get Correlated Features  
```python
def get_correlation(data, threshold):
    corr_col = set()
    corrmat = data.corr()
    for i in range(len(corrmat.columns)):
        for j in range(i):
            if abs(corrmat.iloc[i, j]) > threshold:
                colname = corrmat.columns[i]
                corr_col.add(colname)
    return corr_col
```

### Drop Correlated Features  
```python
corr_features = get_correlation(X_train_unique, 0.85)
X_train_uncorr = X_train_unique.drop(labels=corr_features, axis=1)
X_test_uncorr = X_test_unique.drop(labels=corr_features, axis=1)
```

---

## Feature Grouping and Feature Importance  

### Correlation Matrix (Subset)  
|       | 0       | 1       | 2       | 3       | 4       | ... |
|-------|---------|---------|---------|---------|---------|-----|
| 0     | 1.000000| -0.025277| -0.001942| 0.003594| 0.004054| ... |
| 1     | -0.025277| 1.000000| -0.007647| 0.001819| 0.008981| ... |
| 2     | -0.001942| -0.007647| 1.000000| 0.030919| 0.106245| ... |
| ...   | ...     | ...     | ...     | ...     | ...     | ... |

---

## Feature Importance Based on Tree-Based Classifiers  
```python
important_features = []
for group in correlated_groups_list:
    features = list(group.features1.unique()) + list(group.features2.unique())
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(X_train_unique[features], y_train)
    importance = pd.concat([pd.Series(features), pd.Series(rf.feature_importances_)], axis=1)
    importance.columns = ['features', 'importance']
    importance.sort_values(by='importance', ascending=False, inplace=True)
    feat = importance.iloc[0]
    important_features.append(feat)
```

---

## Final Model Comparison  
```python
run_randomForest(X_train_grouped_uncorr, X_test_grouped_uncorr, y_train, y_test)
# Accuracy: 0.95775
```

---

**Source:** [https://kgptalkie.com/feature-selection-with-filtering-method-constant-quasi-constant-and-duplicate-feature-removal](https://kgptalkie.com/feature-selection-with-filtering-method-constant-quasi-constant-and-duplicate-feature-removal)