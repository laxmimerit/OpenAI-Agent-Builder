# Feature Selection Based on Univariate ROC_AUC for Classification and MSE for Regression | Machine Learning | KGP Talkie  
https://kgptalkie.com/feature-selection-based-on-univariate-roc_auc-for-classification-and-mse-for-regression-machine-learning-kgp-talkie  

Published by **KGP Talkie** on **11 August 2020**  

## Watch Full Playlist  
[https://www.youtube.com/playlist?list=PLc2rvfiptPSQYzmDIFuq2PqN2n28ZjxDH](https://www.youtube.com/playlist?list=PLc2rvfiptPSQYzmDIFuq2PqN2n28ZjxDH)  

---

## What is ROC_AUC  

The **Receiver Operator Characteristic (ROC)** curve is well-known in evaluating classification performance. Owing to its superiority in dealing with imbalanced and cost-sensitive data, the **ROC** curve has been exploited as a popular metric to evaluate **ML** models.  

The **ROC** curve and **AUC** (area under the ROC curve) have been widely used to determine the classification accuracy in supervised learning. It is basically used in **Binary Classification**.  

---

## Use of ROC_AUC in Classification Problem  

### Importing Required Libraries  
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_selection import VarianceThreshold
```

### Download Dataset  
[https://github.com/laxmimerit/Data-Files-for-Feature-Selection](https://github.com/laxmimerit/Data-Files-for-Feature-Selection)  

### Load Data  
```python
data = pd.read_csv('train.csv', nrows=20000)
data.head()
```

**Output:**  
| ID | var3 | var15 | imp_ent_var16_ult1 | ... | TARGET |
|----|------|-------|---------------------|-----|--------|
| 0  | 1    | 2     | 23                  | ... | 0      |
| 1  | 3    | 2     | 34                  | ... | 0      |
| 2  | 4    | 2     | 23                  | ... | 0      |
| 3  | 8    | 2     | 37                  | ... | 0      |
| 4  | 10   | 2     | 39                  | ... | 0      |

### Split Data  
```python
X = data.drop('TARGET', axis=1)
y = data['TARGET']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
```

### Remove Constant, Quasi-Constant, and Duplicate Features  
```python
constant_filter = VarianceThreshold(threshold=0.01)
constant_filter.fit(X_train)
X_train_filter = constant_filter.transform(X_train)
X_test_filter = constant_filter.transform(X_test)
```

**Output:**  
```python
X_train_filter.shape, X_test_filter.shape
((16000, 245), (4000, 245))
```

### Remove Duplicate Features  
```python
X_train_T = X_train_filter.T
X_test_T = X_test_filter.T
X_train_T = pd.DataFrame(X_train_T)
X_test_T = pd.DataFrame(X_test_T)

duplicated_features = X_train_T.duplicated()
features_to_keep = [not index for index in duplicated_features]
X_train_unique = X_train_T[features_to_keep].T
X_test_unique = X_test_T[features_to_keep].T
```

**Output:**  
```python
X_train_unique.shape, X_train.shape
((16000, 227), (16000, 370))
```

---

## Calculate ROC_AUC Score  
```python
roc_auc = []
for feature in X_train_unique.columns:
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train_unique[feature].to_frame(), y_train)
    y_pred = clf.predict(X_test_unique[feature].to_frame())
    roc_auc.append(roc_auc_score(y_test, y_pred))
print(roc_auc)
```

**Output (partial):**  
```python
[0.5020561820568537, 0.5, 0.5, 0.49986968986187125, 0.501373452866903, ...]
```

### Sort ROC_AUC Values  
```python
roc_values = pd.Series(roc_auc)
roc_values.index = X_train_unique.columns
roc_values.sort_values(ascending=False, inplace=True)
```

**Output (partial):**  
```python
244    0.507660
107    0.504832
104    0.502937
...
```

### Select Features with ROC_AUC > 0.5  
```python
sel = roc_values[roc_values > 0.5]
X_train_roc = X_train_unique[sel.index]
X_test_roc = X_test_unique[sel.index]
```

---

## Build Model and Compare Performance  
```python
def run_randomForest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy on test set: ', accuracy_score(y_test, y_pred))
```

### Performance on Selected Features  
```python
%%time
run_randomForest(X_train_roc, X_test_roc, y_train, y_test)
```

**Output:**  
```
Accuracy on test set:  0.95275
Wall time: 917 ms
```

### Performance on Original Features  
```python
%%time
run_randomForest(X_train, X_test, y_train, y_test)
```

**Output:**  
```
Accuracy on test set:  0.9585
Wall time: 1.76 s
```

---

## Feature Selection Using RMSE in Regression  

### Load Boston Dataset  
```python
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target
```

### Split Data  
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

### Calculate MSE for Each Feature  
```python
mse = []
for feature in X_train.columns:
    clf = LinearRegression()
    clf.fit(X_train[feature].to_frame(), y_train)
    y_pred = clf.predict(X_test[feature].to_frame())
    mse.append(mean_squared_error(y_test, y_pred))
```

**Output (partial):**  
```python
[76.38674157646072, 84.66034377707905, 77.02905244667242, ...]
```

### Sort MSE Values  
```python
mse = pd.Series(mse, index=X_train.columns)
mse.sort_values(ascending=False, inplace=True)
```

**Output (partial):**  
```python
ZN         84.660344
DIS        82.618741
RAD        82.465000
...
```

### Select Top 2 Features  
```python
X_train_2 = X_train[['RM', 'LSTAT']]
X_test_2 = X_test[['RM', 'LSTAT']]
```

### Model Performance on Selected Features  
```python
model = LinearRegression()
model.fit(X_train_2, y_train)
y_pred = model.predict(X_test_2)
print('r2_score: ', r2_score(y_test, y_pred))
print('rmse: ', np.sqrt(mean_squared_error(y_test, y_pred)))
```

**Output:**  
```
r2_score:  0.5409084827186417
rmse:  6.114172522817782
```

### Model Performance on Original Features  
```python
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('r2_score: ', r2_score(y_test, y_pred))
print('rmse: ', np.sqrt(mean_squared_error(y_test, y_pred)))
```

**Output:**  
```
r2_score:  0.5892223849182507
rmse:  5.783509315085135
```

---

## References  
- [https://kgptalkie.com/feature-selection-based-on-univariate-roc_auc-for-classification-and-mse-for-regression-machine-learning-kgp-talkie](https://kgptalkie.com/feature-selection-based-on-univariate-roc_auc-for-classification-and-mse-for-regression-machine-learning-kgp-talkie)  
- [https://archive.ics.uci.edu/ml/machine-learning-databases/housing/](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/)