https://kgptalkie.com/feature-dimention-reduction-using-lda-and-pca-with-python-principal-component-analysis-in-feature-selection-kgp-talkie

# Feature Dimension Reduction Using LDA and PCA with Python  
**Principal Component Analysis in Feature Selection | KGP Talkie**  

Published by  
**KGP Talkie**  
on  
**10 August 2020**  

---

## Feature Dimension Reduction  
Watch Full Playlist:  
https://www.youtube.com/playlist?list=PLc2rvfiptPSQYzmDIFuq2PqN2n28ZjxDH  

---

## What is LDA (Linear Discriminant Analysis)?  
The idea behind **LDA** is simple. Mathematically speaking, we need to find a new feature space to project the data in order to **maximize classes separability**.  

### Key Concepts:  
- **Supervised Algorithm**: Takes class labels into consideration.  
- **Dimensionality Reduction**: Preserves class discrimination information.  
- **Cluster Boundaries**: Projects data points on a line to separate clusters with minimal intra-cluster variation.  

### How LDA Works:  
1. **Centroid Calculation**: Finds the centroid of each class using the dataset's features.  
2. **Dimension Selection**: Determines a new dimension (axis) that:  
   - **Maximizes** the distance between centroids of each class.  
   - **Minimizes** the variation (scatter, `s²`) within each category.  

---

## What is PCA (Principal Component Analysis)?  
**PCA** is a linear dimensionality reduction technique that projects high-dimensional data into a lower-dimensional subspace while preserving essential variations.  

### Key Concepts:  
- **Unsupervised**: No labels required; clusters data based on feature correlation.  
- **Orthogonal Transformation**: Converts correlated variables into uncorrelated principal components.  
- **Use Cases**:  
  - **Data Visualization**: Simplifies high-dimensional data for analysis.  
  - **Speeding ML Algorithms**: Reduces training/testing time by reducing feature count.  

---

## When to Use PCA  
### 1. **Data Visualization**  
- Challenge: High-dimensional data with many variables.  
- Solution: PCA reduces dimensions for easier visualization.  

### 2. **Speeding Machine Learning (ML) Algorithms**  
- Benefit: Reduces computational complexity by eliminating redundant features.  

---

## How to Do PCA  
### Steps:  
1. **Import Libraries**:  
   ```python
   import pandas as pd
   import numpy as np
   import seaborn as sns
   import matplotlib.pyplot as plt
   %matplotlib inline
   from sklearn.model_selection import train_test_split
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import accuracy_score, roc_auc_score
   from sklearn.feature_selection import VarianceThreshold
   from sklearn.preprocessing import StandardScaler
   ```

2. **Load Data**:  
   ```python
   data = pd.read_csv('santander.csv', nrows=20000)
   X = data.drop('TARGET', axis=1)
   y = data['TARGET']
   X.shape, y.shape  # ((20000, 370), (20000,))
   ```

3. **Train-Test Split**:  
   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
   ```

4. **Remove Constant/Quasi-Constant Features**:  
   ```python
   constant_filter = VarianceThreshold(threshold=0.01)
   constant_filter.fit(X_train)
   X_train_filter = constant_filter.transform(X_train)
   X_test_filter = constant_filter.transform(X_test)
   ```

5. **Remove Duplicates**:  
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

6. **Standardize Data**:  
   ```python
   scaler = StandardScaler().fit(X_train_unique)
   X_train_unique = scaler.transform(X_train_unique)
   X_test_unique = scaler.transform(X_test_unique)
   ```

7. **Remove Correlated Features**:  
   ```python
   corrmat = X_train_unique.corr()
   def get_correlation(data, threshold):
       corr_col = set()
       corrmat = data.corr()
       for i in range(len(corrmat.columns)):
           for j in range(i):
               if abs(corrmat.iloc[i, j]) > threshold:
                   colname = corrmat.columns[i]
                   corr_col.add(colname)
       return corr_col
   corr_features = get_correlation(X_train_unique, 0.70)
   X_train_uncorr = X_train_unique.drop(labels=corr_features, axis=1)
   X_test_uncorr = X_test_unique.drop(labels=corr_features, axis=1)
   ```

---

## Feature Dimension Reduction by LDA  
### Code:  
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=1)
X_train_lda = lda.fit_transform(X_train_uncorr, y_train)
X_test_lda = lda.transform(X_test_uncorr)
```

### Performance:  
```python
def run_randomForest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy on test set: ')
    print(accuracy_score(y_test, y_pred))
```

**Results**:  
- **LDA-Transformed Data**: Accuracy = 0.93025, CPU Time = 1.1s  
- **Uncorrelated Data**: Accuracy = 0.9585, CPU Time = 1.2s  
- **Original Data**: Accuracy = 0.9585, CPU Time = 2.01s  

---

## Feature Reduction by PCA  
### Code:  
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=42)
pca.fit(X_train_uncorr)
X_train_pca = pca.transform(X_train_uncorr)
X_test_pca = pca.transform(X_test_uncorr)
```

### Performance:  
- **PCA-Transformed Data (2 components)**: Accuracy = 0.956, CPU Time = 999ms  
- **Original Data**: Accuracy = 0.9585, CPU Time = 2.13s  

### Component Analysis:  
```python
for component in range(1, 5):
    pca = PCA(n_components=component, random_state=42)
    pca.fit(X_train_uncorr)
    X_train_pca = pca.transform(X_train_uncorr)
    X_test_pca = pca.transform(X_test_uncorr)
    print(f'Selected Components: {component}')
    run_randomForest(X_train_pca, X_test_pca, y_train, y_test)
```

**Results**:  
- **1 Component**: Accuracy = 0.92375  
- **2 Components**: Accuracy = 0.956  
- **3 Components**: Accuracy = 0.95675  
- **4 Components**: Accuracy = 0.95825  

---

**Source:** https://kgptalkie.com/feature-dimention-reduction-using-lda-and-pca-with-python-principal-component-analysis-in-feature-selection-kgp-talkie