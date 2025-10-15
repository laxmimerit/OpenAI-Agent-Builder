# Step Forward, Step Backward and Exhaustive Feature Selection | Wrapper Method | KGP Talkie  
**Source:** https://kgptalkie.com/step-forward-step-backward-and-exhaustive-feature-selection-wrapper-method-kgp-talkie  

Published by  
**KGP Talkie**  
on  
**9 August 2020**  

---

## Wrapping Method  

### Uses of Wrapping Method  
- Uses combinations of variables to determine predictive power.  
- To find the best combination of variables.  
- Computationally expensive than filter method.  
- To perform better than filter method.  
- Not recommended on high number of features.  

---

## Forward Step Selection  

In this wrapping method, it selects one **best feature** every time and finally it **combines** all the best features for the best **accuracy**.  

---

## Backward Step Selection  

It is the **reverse** process of **Forward Step Selection** method. Initially, it takes **all** the features and **remove** one by one every time. Finally, it left with required number of features for the best **accuracy**.  

---

## Exhaustive Feature Selection  

It is also called as subset selection method. It fits the model with each possible combination of N features:  
- $ y = B0 $  
- $ y = B0 + B1.X1 $  
- $ y = C0 + C1.X2 $  

It requires massive computational power. It uses test error to evaluate model performance.  

### Drawback  
It is a **slower** method compared to step forward and backward methods.  

---

## Use of mlxtend in Wrapper Method  

```bash
pip install mlxtend
```

**Output:**  
```
Requirement already satisfied: mlxtend in c:\users\srish\appdata\roaming\python\python38\site-packages (0.17.3)
Requirement already satisfied: scikit-learn>=0.20.3 in e:\callme_conda\lib\site-packages (from mlxtend) (0.23.1)
...
```

**More Information Available at:**  
http://rasbt.github.io/mlxtend/  

---

## How it Works  

Sequential feature selection algorithms are a family of **greedy search algorithms** that are used to reduce an initial **d-dimensional** feature space to a **k-dimensional** feature subspace where **k < d**.  

In a nutshell, **SFAs** remove or add one feature at a time based on the classifier **performance** until a feature subset of the desired size **k** is reached. There are **4** different flavors of **SFAs** available via the SequentialFeatureSelector:  
- Sequential Forward Selection (SFS)  
- Sequential Backward Selection (SBS)  
- Sequential Forward Floating Selection (SFFS)  
- Sequential Backward Floating Selection (SBFS)  

---

## Step Forward Selection (SFS)  

### Importing Required Libraries  

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
```

### Loading the Wine Dataset  

```python
data = load_wine()
```

### Dataset Keys  

```python
data.keys()
```

**Output:**  
```
dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names'])
```

### Dataset Description  

```python
print(data.DESCR)
```

**Source:** https://kgptalkie.com/step-forward-step-backward-and-exhaustive-feature-selection-wrapper-method-kgp-talkie  

---

### Data Preparation  

```python
X = pd.DataFrame(data.data)
y = data.target
X.columns = data.feature_names
X.head()
```

### Checking for Null Values  

```python
X.isnull().sum()
```

**Output:**  
```
alcohol                         0
malic_acid                      0
ash                             0
alcalinity_of_ash               0
magnesium                       0
...
```

### Train-Test Split  

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train.shape, X_test.shape
```

**Output:**  
```
((142, 13), (36, 13))
```

### Sequential Feature Selection (SFS)  

```python
sfs = SFS(RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1),
          k_features=7,
          forward=True,
          floating=False,
          verbose=2,
          scoring='accuracy',
          cv=4,
          n_jobs=-1
         ).fit(X_train, y_train)
```

**Output:**  
```
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
...
```

### Selected Features  

```python
sfs.k_feature_names_
```

**Output:**  
```
('alcohol', 'ash', 'magnesium', 'flavanoids', 'proanthocyanins', 'color_intensity', 'proline')
```

---

## Step Backward Selection (SBS)  

### Sequential Feature Selection (SBS)  

```python
sfs = SFS(RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1),
          k_features=(1, 8),
          forward=False,
          floating=False,
          verbose=2,
          scoring='accuracy',
          cv=4,
          n_jobs=-1
         ).fit(X_train, y_train)
```

**Output:**  
```
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
...
```

### Selected Features  

```python
sbs.k_feature_names_
```

**Output:**  
```
('alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'flavanoids', 'nonflavanoid_phenols', 'color_intensity')
```

---

## Exhaustive Feature Selection (EFS)  

### Exhaustive Feature Selection  

```python
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

efs = EFS(RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1),
          min_features=4,
          max_features=5,
          scoring='accuracy',
          cv=None,
          n_jobs=-1
         ).fit(X_train, y_train)
```

**Output:**  
```
Features: 2002/2002
```

### Best Accuracy  

```python
efs.best_score_
```

**Output:**  
```
1.0
```

### Selected Features  

```python
efs.best_feature_names_
```

**Output:**  
```
('alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash')
```

---

## Plotting Performance  

```python
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

plot_sfs(efs.get_metric_dict(), kind='std_dev')
plt.title('Performance of the EFS algorithm with changing number of features')
plt.show()
```

**Source:** https://kgptalkie.com/step-forward-step-backward-and-exhaustive-feature-selection-wrapper-method-kgp-talkie