https://kgptalkie.com/pca-with-python-principal-component-analysis-machine-learning-kgp-talkie%C2%B6

# PCA with Python | Principal Component Analysis Machine Learning | KGP Talkie

Published by  
**KGP Talkie**  
on  
**8 August 2020**  
**8 August 2020**

## Principal Component Analysis (PCA)

According to Wikipedia, PCA is a statistical procedure that uses an **orthogonal transformation** to convert a set of observations of possibly **correlated variables** (entities each of which takes on various numerical values) into a set of values of linearly uncorrelated variables called **principal components**.

### Principal Components

These are the new axes that describe the **variation** in the data.

- **Principal component 1**: The axis which spans the **most variation** of the data.
- **Principal component 2**: The axis which spans the **second most variation** of the data.
- **Principal component 3**: The axis which spans the **third most variation** of the data and so on.

## When to Use PCA

We can use PCA in the following cases:

- **Data Visualization**: It is used to find **inter-relation** between variables in the data.
- **Speeding Machine Learning (ML) Algorithm**: It’s often used to visualize **genetic distance** and **relatedness** between populations.
- **Simplifying Analysis**: As the number of variables decreases, it makes further analysis simpler.

## Objectives of PCA

The main objectives of the PCA are:

- It is basically a **non-dependent** procedure in which it reduces attribute space from a large number of variables to a smaller number of factors.
- PCA is basically a **dimension reduction** process but there is no guarantee that the **dimension** is interpretable.
- The main task in this PCA is to select a subset of variables from a larger set, based on which original variables have the highest **correlation** with the principal amount.

## How to Do PCA

As there are as many **principal components** as there are variables in the data, principal components are constructed in such a manner that:

- The first principal component accounts for the largest possible **variance** in the data set.
- The second principal component is calculated in the same way, with the condition that it is **uncorrelated** with (i.e., perpendicular to) the first principal component and that it accounts for the next **highest variance**.

Once fit, the **eigenvalues** and principal components can be accessed on the PCA class via the **explained_variance_** and **components_ attributes**.

### Principal Axis Method

PCA will search a **linear combination** of variables so that we can extract **maximum variance** from the variables. Once this process completes, it will remove it and search for another **linear combination** which will give an explanation about the maximum proportion of remaining **variance**, which basically leads to **orthogonal** factors. In this method, we analyze total **variance**.

## PCA Summary

From the following figure, I will make you understand how **PCA** works in a nutshell.

There are a few steps. Let’s see one after another:

1. In the first step, we have a **correlated high dimension** data. And then we calculate the **center** of the points and calculate **variance** of the data by using **covariance matrix** of the data. With this matrix, we calculate **eigen vectors** and **eigen values**.
2. After calculating these, we pick the value of **m** such that less than the original dimension.
3. Then, after we will project **data points into those eigen vectors** and we do the inverse transform so we will get **uncorrelated low dimensional** data.

Though mathematically it looks a little bit complex, but fortunately in Python we have the **sklearn library**, there we have **PCA** package. Just we call **PCA()** and then call **pca.fit()** as usual we do in ML algorithms.

## Importing Required Libraries

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

## Loading the Training Data Set

```python
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
cancer = datasets.load_breast_cancer()
```

Let’s go ahead and get the description of the **breast cancer** data set.

```python
print(cancer.DESCR)
```

### Breast Cancer Wisconsin (Diagnostic) Dataset

**Data Set Characteristics:**

- **Number of Instances**: 569
- **Number of Attributes**: 30 numeric, predictive attributes and the class

**Source:** https://kgptalkie.com/pca-with-python-principal-component-analysis-machine-learning-kgp-talkie%C2%B6

**Attribute Information:**

- **radius**: Mean of distances from center to points on the perimeter
- **texture**: Standard deviation of gray-scale values
- **perimeter**
- **area**
- **smoothness**: Local variation in radius lengths
- **compactness**: Perimeter^2 / area - 1.0
- **concavity**: Severity of concave portions of the contour
- **concave points**: Number of concave portions of the contour
- **symmetry**
- **fractal dimension**: "Coastline approximation" - 1

The mean, standard error, and "worst" or largest (mean of the three worst/largest values) of these features were computed for each image, resulting in 30 features. For instance, field 0 is **Mean Radius**, field 10 is **Radius SE**, field 20 is **Worst Radius**.

- **Class**:
  - **WDBC-Malignant**
  - **WDBC-Benign**

**Summary Statistics:**

| Attribute | Min | Max |
|---------|-----|-----|
| radius (mean) | 6.981 | 28.11 |
| texture (mean) | 9.71 | 39.28 |
| perimeter (mean) | 43.79 | 188.5 |
| area (mean) | 143.5 | 2501.0 |
| smoothness (mean) | 0.053 | 0.163 |
| compactness (mean) | 0.019 | 0.345 |
| concavity (mean) | 0.0 | 0.427 |
| concave points (mean) | 0.0 | 0.201 |
| symmetry (mean) | 0.106 | 0.304 |
| fractal dimension (mean) | 0.05 | 0.097 |
| radius (standard error) | 0.112 | 2.873 |
| texture (standard error) | 0.36 | 4.885 |
| perimeter (standard error) | 0.757 | 21.98 |
| area (standard error) | 6.802 | 542.2 |
| smoothness (standard error) | 0.002 | 0.031 |
| compactness (standard error) | 0.002 | 0.135 |
| concavity (standard error) | 0.0 | 0.396 |
| concave points (standard error) | 0.0 | 0.053 |
| symmetry (standard error) | 0.008 | 0.079 |
| fractal dimension (standard error) | 0.001 | 0.03 |
| radius (worst) | 7.93 | 36.04 |
| texture (worst) | 12.02 | 49.54 |
| perimeter (worst) | 50.41 | 251.2 |
| area (worst) | 185.2 | 4254.0 |
| smoothness (worst) | 0.071 | 0.223 |
| compactness (worst) | 0.027 | 1.058 |
| concavity (worst) | 0.0 | 1.252 |
| concave points (worst) | 0.0 | 0.291 |
| symmetry (worst) | 0.156 | 0.664 |
| fractal dimension (worst) | 0.055 | 0.208 |

**Source:** https://kgptalkie.com/pca-with-python-principal-component-analysis-machine-learning-kgp-talkie%C2%B6

**Missing Attribute Values:** None  
**Class Distribution:** 212 - Malignant, 357 - Benign  
**Creator:** Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian  
**Donor:** Nick Street  
**Date:** November, 1995

**Source:** https://kgptalkie.com/pca-with-python-principal-component-analysis-machine-learning-kgp-talkie%C2%B6

This is a copy of UCI ML Breast Cancer Wisconsin (Diagnostic) datasets.  
https://goo.gl/U2Uwz2

Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.

Separating plane described above was obtained using **Multisurface Method-Tree (MSM-T)** [K. P. Bennett, "Decision Tree Construction Via Linear Programming." Proceedings of the 4th Midwest Artificial Intelligence and Cognitive Science Society, pp. 97-101, 1992], a classification method which uses linear programming to construct a decision tree. Relevant features were selected using an exhaustive search in the space of 1-4 features and 1-3 separating planes.

The actual linear program used to obtain the separating plane in the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].

This database is also available through the UW CS ftp server:

```
ftp ftp.cs.wisc.edu
cd math-prog/cpo-dataset/machine-learn/WDBC/
```

This data has **30-dimensions** that is **30 features**. Let’s visualize the data set with **dataframe**.

```python
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df.head()
```

| mean radius | mean texture | mean perimeter | mean area | mean smoothness | mean compactness | mean concavity | mean concave points | mean symmetry | mean fractal dimension | ... | worst radius | worst texture | worst perimeter | worst area | worst smoothness | worst compactness | worst concavity | worst concave points | worst symmetry | worst fractal dimension |
|-------------|--------------|----------------|-----------|------------------|------------------|----------------|---------------------|----------------|------------------------|-----|--------------|----------------|------------------|-------------|------------------|-------------------|------------------|----------------------|----------------|--------------------------|
| 17.99       | 10.38        | 122.80         | 1001.0    | 0.11840          | 0.27760          | 0.3001         | 0.14710             | 0.2419         | 0.07871                | ... | 25.38        | 17.33          | 184.60           | 2019.0      | 0.1622           | 0.6656            | 0.7119           | 0.2654               | 0.4601         | 0.11890                  |
| 20.57       | 17.77        | 132.90         | 1326.0    | 0.08474          | 0.07864          | 0.0869         | 0.07017             | 0.1812         | 0.05667                | ... | 24.99        | 23.41          | 158.80           | 1956.0      | 0.1238           | 0.1866            | 0.2416           | 0.1860               | 0.2750         | 0.08902                  |
| 19.69       | 21.25        | 130.00         | 1203.0    | 0.10960          | 0.15990          | 0.1974         | 0.12790             | 0.2069         | 0.05999                | ... | 23.57        | 25.53          | 152.50           | 1709.0      | 0.1444           | 0.4245            | 0.4504           | 0.2430               | 0.3613         | 0.08758                  |
| 11.42       | 20.38        | 77.58          | 386.1     | 0.14250          | 0.28390          | 0.2414         | 0.10520             | 0.2597         | 0.09744                | ... | 14.91        | 26.50          | 98.87            | 567.7       | 0.2098           | 0.8663            | 0.6869           | 0.2575               | 0.6638         | 0.17300                  |
| 20.29       | 14.34        | 135.10         | 1297.0    | 0.10030          | 0.13280          | 0.1980         | 0.10430             | 0.1809         | 0.05883                | ... | 22.54        | 16.67          | 152.20           | 1575.0      | 0.1374           | 0.2050            | 0.4000           | 0.1625               | 0.2364         | 0.07678                  |

5 rows × 30 columns

If we see here, the scale of each feature is different. Some features are in the range **10s**, some are in **100s**. It is better to **standardize** our data for better visualization.

Let’s see the below code:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
X_scaled[:2]
```

```python
array([[ 1.09706398e+00, -2.07333501e+00,  1.26993369e+00,
         9.84374905e-01,  1.56846633e+00,  3.28351467e+00,
         2.65287398e+00,  2.53247522e+00,  2.21751501e+00,
         2.25574689e+00,  2.48973393e+00, -5.65265059e-01,
         2.83303087e+00,  2.48757756e+00, -2.14001647e-01,
         1.31686157e+00,  7.24026158e-01,  6.60819941e-01,
         1.14875667e+00,  9.07083081e-01,  1.88668963e+00,
        -1.35929347e+00,  2.30360062e+00,  2.00123749e+00,
         1.30768627e+00,  2.61666502e+00,  2.10952635e+00,
         2.29607613e+00,  2.75062224e+00,  1.93701461e+00],
       [ 1.82982061e+00, -3.53632408e-01,  1.68595471e+00,
         1.90870825e+00, -8.26962447e-01, -4.87071673e-01,
        -2.38458552e-02,  5.48144156e-01,  1.39236330e-03,
        -8.68652457e-01,  4.99254601e-01, -8.76243603e-01,
         2.63326966e-01,  7.42401948e-01, -6.05350847e-01,
        -6.92926270e-01, -4.40780058e-01,  2.60162067e-01,
        -8.05450380e-01, -9.94437403e-02,  1.80592744e+00,
        -3.69203222e-01,  1.53512599e+00,  1.89048899e+00,
        -3.75611957e-01, -4.30444219e-01, -1.46748968e-01,
         1.08708430e+00, -2.43889668e-01,  2.81189987e-01]])
```

## PCA()

**Principal Component Analysis**

**Linear dimensionality reduction** using **Singular Value Decomposition** of the data to project it to a lower dimensional space. The input data is centered but not scaled for each feature before applying the **SVD**.

Applying the **PCA()** function into training and testing set for analysis.

Let’s discuss some important parameters of the function **PCA()**:

- **n_components**: It is a int, float, None or str. Number of components to keep.
- **random_state**: It is to pass an **int** for reproducible results across multiple function calls.
- **explained_variance**: It is provide the amount of **variance** explained by each of the selected components.
- **fit_transform**: It helps to fit the model with X and apply the **dimensionality reduction** on X.
- **inverse_transform**: It transforms data **back** to its **original** space.

We need to have **2-dimensional** data set so **n_component** is equal to **2** and we can get the same result **random_state** if we use the same **random_state**.

Applying the **PCA** function into training and testing set for analysis look at the following code:

### Fit the Model X_scaled by Using `pac.fit()`:

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=42)
pca.fit(X_scaled)
PCA(n_components=2, random_state=42)
```

Training has been done now let’s go ahead to transform it with **pca.transform()**:

```python
X_pca = pca.transform(X_scaled)
```

So we transformed the data into **2-dimensions**.

Let’s check the **shape** of both data sets:

```python
X_scaled.shape, X_pca.shape
```

```python
((569, 30), (569, 2))
```

Here we can observe the shape of the default data is **30** and after transformation, it reduced to **2**.

Now we will try to plot the **scattering points** for the second principal component and first principal component by using **plt.scatter()** function.

Let’s look into the following script:

```python
plt.figure(figsize=(12,8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c = cancer.target, cmap = 'viridis')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Scatter plot for Second principal component and First principal component')
plt.show()
```

From the above plot, we can observe, first principal component has **high variance** compared to the second principal component.

```python
pca.explained_variance_ratio_
```

```python
array([0.44272026, 0.18971182])
```

Now we will observe the respective **variances** for the components by using a bar graph.

See the following plot:

```python
pca = PCA(n_components=20, random_state=42)
X_pca = pca.fit_transform(X_scaled)
variance = pca.explained_variance_ratio_
plt.ylabel('Variance')
plt.xlabel('Principal components')
plt.title('Bar graph for variances of the components ')
plt.bar(x = range(1, len(variance)+1), height=variance, width=0.7)
plt.show()
```

```python
variance
array([0.44272026, 0.18971182, 0.09393163, 0.06602135, 0.05495768,
0.04024522, 0.02250734, 0.01588724, 0.01389649, 0.01168978,
0.00979719, 0.00870538, 0.00804525, 0.00523366, 0.00313783,
0.00266209, 0.00197997, 0.00175396, 0.00164925, 0.00103865])
```