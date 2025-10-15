https://kgptalkie.com/recursive-feature-elimination-rfe-by-using-tree-based-and-gradient-based-estimators-machine-learning-kgp-talkie

# Recursive Feature Elimination (RFE) by Using Tree Based and Gradient Based Estimators | Machine Learning | KGP Talkie

**Published by**: KGP Talkie  
**Date**: 10 August 2020

## Introduction to RFE

Recursive Feature Elimination (RFE) is a feature selection method that recursively removes the least important features based on model performance. It works by:

1. Training a model on all features
2. Calculating feature importance
3. Removing the least important feature
4. Repeating the process until the desired number of features is reached

## Implementation with Scikit-Learn

### Required Libraries

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.metrics import accuracy_score
```

### Dataset Preparation

```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
```

**Dataset Description**:
- **Number of Instances**: 569
- **Number of Attributes**: 30 numeric predictive attributes
- **Class Distribution**: 212 Malignant, 357 Benign

**Source**: [Breast Cancer Wisconsin Dataset](https://goo.gl/U2Uwz2)

### Data Loading

```python
X = pd.DataFrame(data=data.data, columns=data.feature_names)
y = data.target
```

### Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train.shape, X_test.shape  # ((455, 30), (114, 30))
```

## Feature Selection Methods

### 1. Random Forest Classifier with SelectFromModel

```python
sel = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1))
sel.fit(X_train, y_train)
```

**Selected Features**:
- `mean radius`
- `mean perimeter`
- `mean area`
- `mean concavity`
- `mean concave points`
- `area error`
- `worst radius`
- `worst perimeter`
- `worst area`
- `worst concave points`

**Accuracy with 10 features**: 94.74%  
**Accuracy with full dataset**: 96.49% (2% decrease)

### 2. Recursive Feature Elimination (RFE)

#### With Random Forest
```python
sel = RFE(RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1), n_features_to_select=15)
sel.fit(X_train, y_train)
```

**Selected Features** (15 features):  
- `mean radius`, `mean texture`, `mean perimeter`, `mean area`, `mean concavity`, `mean concave points`, `area error`, `worst radius`, `worst texture`, `worst perimeter`, `worst area`, `worst smoothness`, `worst concavity`, `worst concave points`, `worst symmetry`

**Accuracy with 15 features**: 97.37%  
**Accuracy with full dataset**: 96.49% (1% increase)

#### With Gradient Boosting
```python
sel = RFE(GradientBoostingClassifier(n_estimators=100, random_state=0), n_features_to_select=12)
sel.fit(X_train, y_train)
```

**Selected Features** (12 features):  
- `mean texture`, `mean smoothness`, `mean concave points`, `mean symmetry`, `area error`, `concavity error`, `worst radius`, `worst texture`, `worst perimeter`, `worst area`, `worst concavity`, `worst concave points`

**Accuracy with 12 features**: 97.37%  
**Accuracy with full dataset**: 96.49% (1% increase)

## Accuracy Comparison with Varying Features

### Gradient Boosting
```python
for index in range(1, 31):
    sel = RFE(GradientBoostingClassifier(n_estimators=100, random_state=0), n_features_to_select=index)
    sel.fit(X_train, y_train)
    X_train_rfe = sel.transform(X_train)
    X_test_rfe = sel.transform(X_test)
    print(f'Selected Feature: {index}')
    run_randomForest(X_train_rfe, X_test_rfe, y_train, y_test)
    print()
```

**Key Results**:
- **6 features**: 99.12% accuracy
- **12 features**: 97.37% accuracy
- **15 features**: 97.37% accuracy

### Random Forest
```python
for index in range(1, 31):
    sel = RFE(RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1), n_features_to_select=index)
    sel.fit(X_train, y_train)
    X_train_rfe = sel.transform(X_train)
    X_test_rfe = sel.transform(X_test)
    print(f'Selected Feature: {index}')
    run_randomForest(X_train_rfe, X_test_rfe, y_train, y_test)
    print()
```

**Key Results**:
- **17 features**: 98.25% accuracy
- **24 features**: 98.25% accuracy
- **28 features**: 96.49% accuracy

## Conclusion

- **Optimal Feature Count**: 6 features (Gradient Boosting) achieved 99.12% accuracy
- **Selected Features**: `mean concave points`, `area error`, `worst texture`, `worst perimeter`, `worst area`, `worst concave points`
- **Performance**: Significant accuracy improvements with fewer features

**Source**: [https://kgptalkie.com/recursive-feature-elimination-rfe-by-using-tree-based-and-gradient-based-estimators-machine-learning-kgp-talkie](https://kgptalkie.com/recursive-feature-elimination-rfe-by-using-tree-based-and-gradient-based-estimators-machine-learning-kgp-talkie)