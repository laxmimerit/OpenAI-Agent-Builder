# K-Mean Clustering in Python | Machine Learning | KGP Talkie  
**Source:** https://kgptalkie.com/k-mean-clustering-in-python-machine-learning-kgp-talkie  

## Published by  
**KGP Talkie**  
**on 7 August 2020**  

---

## What is K-Mean Clustering?

Machine Learning can broadly be classified into three types:

- **Supervised Learning**
- **Unsupervised Learning**
- **Semi-supervised Learning**

The **K-means algorithm** identifies *k* number of centroids and allocates every data point to the nearest cluster while keeping the centroids as small as possible. The term **"means"** in K-means refers to the **averaging of the data**, i.e., finding the centroid.

---

## Types of Clustering

### Hard Clustering
- Data points belong exclusively to one cluster.

### Soft Clustering
- Data points can belong to multiple clusters with varying degrees of membership.

---

## Type of Clustering Algorithms

### Connectivity-based Clustering
- Based on the idea that data points closer in the data space are more related (similar) than those farther away.
- **Drawback:** Not robust to outliers, which may form additional clusters or cause merges.

### Centroid-based Clustering
- Clusters are represented by a central **vector** or **centroid**.
- The centroid may not necessarily be a member of the dataset.

### Distribution-based Clustering
- Strong theoretical foundation but prone to overfitting.
- Example: **Gaussian Mixture Models** using the **Expectation-Maximization algorithm**.

### Density-based Methods
- Search for areas of varied density in the data space.

---

## Dataset and Problem Understanding

### Libraries Used
- **Pandas**: For data manipulation.
- **Seaborn** and **Matplotlib**: For data visualization.
- **NumPy**: For array operations.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```

### Fetching the Data
```python
data = pd.read_csv('data.csv', index_col=0)
data.head()
```

|   | x        | y        | cluster |
|---|----------|----------|---------|
| 0 | -8.48285 | -5.60335 | 2       |
| 1 | -7.75163 | -8.40533 | 2       |
| 2 | -10.9671 | -9.03278 | 2       |
| 3 | -11.9994 | -7.60673 | 2       |
| 4 | -1.73681 | 10.47802 | 1       |

### Cluster Distribution
```python
data['cluster'].value_counts()
```

```
1    67
0    67
2    66
Name: cluster, dtype: int64
```

### Visualizing the Dataset
```python
plt.scatter(data['x'], data['y'], c=data['cluster'], cmap='viridis')
plt.xlabel('X-values')
plt.ylabel('Y-values')
plt.title('Formation of cluster')
plt.show()
```

---

## K-Means for Clustering

### Algorithm Overview
- **Unsupervised machine learning technique**.
- Quickly clusters data by finding centroids and assigning points to the nearest cluster.

### Implementation Steps

1. **Import Libraries**
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
```

2. **Standardize the Data**
```python
X = data[['x', 'y']]
y = data['cluster']
scaler = StandardScaler()
X = scaler.fit_transform(X)
```

3. **Train the Model**
```python
k = 2
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)
```

4. **View Centroids**
```python
center = kmeans.cluster_centers_
print(center)
```

```
[[-1.30618271 -0.87560626]
 [ 0.64334372  0.43126875]]
```

5. **Plot Results**
```python
plt.scatter(data['x'], data['y'], c=kmeans.labels_, cmap='viridis')
plt.xlabel('X-values')
plt.ylabel('Y-values')
plt.title('Formation of cluster with centroids')
for i, point in enumerate(center):
    plt.plot(center[i][0], center[i][1], '*r--', linewidth=2, markersize=18)
plt.show()
```

---

## Choosing the Right Value of K

### Elbow Method
- **Error Sum of Squares (SSE)**: Measures variation within clusters.
- **Formula**:
  $$
  \text{SSE} = \sum_{i=1}^k\left(\sum_{x_j \subset S_i}||x_j-\mu_i||^2\right)
  $$

### Code for Elbow Method
```python
SSE = []
index = range(1, 10)
for i in index:
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    SSE.append(kmeans.inertia_)
    print(kmeans.inertia_)
```

### Plotting SSE vs. K
```python
plt.plot(index, SSE)
plt.xlabel('K')
plt.ylabel('SSE')
plt.title('SSE with respect to K')
plt.show()
```

**Source:** https://kgptalkie.com/k-mean-clustering-in-python-machine-learning-kgp-talkie  

---

## Use Iris Dataset

### Load and Preprocess Data
```python
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(iris.target_names)
```

```
['setosa' 'versicolor' 'virginica']
```

### Apply Elbow Method
```python
SSE = []
index = range(1, 10)
for i in index:
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    SSE.append(kmeans.inertia_)
    print(kmeans.inertia_)
```

### Plot Results
```python
plt.plot(index, SSE)
plt.xlabel('K')
plt.ylabel('SSE')
plt.title('Variation of SSE with respect to K')
plt.show()
```

**Source:** https://kgptalkie.com/k-mean-clustering-in-python-machine-learning-kgp-talkie