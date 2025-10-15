https://kgptalkie.com/feature-engineering-tutorial-series-4-linear-model-assumptions

# Feature Engineering Tutorial Series 4: Linear Model Assumptions

Published by [georgiannacambel](https://kgptalkie.com/feature-engineering-tutorial-series-4-linear-model-assumptions) on 2 October 2020

Linear models make the following assumptions over the independent variables **X**, used to predict **Y**:

- There is a linear relationship between **X** and the outcome **Y**
- The independent variables **X** are normally distributed
- There is no or little co-linearity among the independent variables
- Homoscedasticity (homogeneity of variance)

Examples of linear models are:

- Linear and Logistic Regression
- Linear Discriminant Analysis (LDA)
- Principal Component Regressors

## Definitions

### Linear relationship  
describes a relationship between the independent variables **X** and the target **Y** that is given by: **Y ≈ β₀ + β₁X₁ + β₂X₂ + … + βₙXₙ**.

### Normality  
means that every variable **X** follows a Gaussian distribution.

### Multi-collinearity  
refers to the correlation of one independent variable with another. Variables should not be correlated.

### Homoscedasticity  
, also known as homogeneity of variance, describes a situation in which the error term (that is, the “noise” or random disturbance in the relationship between the independent variables **X** and the dependent variable **Y**) is the same across all the independent variables.

**Failure to meet one or more of the model assumptions may end up in a poor model performance.** If the assumptions are not met, we can try a different [machine learning](https://kgptalkie.com/feature-engineering-tutorial-series-4-linear-model-assumptions) model or transform the input variables so that they fulfill the assumptions.

## How can we evaluate if the assumptions are met by the variables?

| Assumption          | Evaluation Method         |
|---------------------|---------------------------|
| Linear relationship | Scatter-plots and residuals plots |
| Normal distribution | Q-Q plots                 |
| Multi-collinearity  | Correlation matrices      |
| Homoscedasticity    | Residuals plots           |

## What can we do if the assumptions are not met?

Sometimes variable transformation can help the variables meet the model assumptions. We normally do 1 of 2 things:

1. **Mathematical transformation of the variables**
2. **Discretisation**

## In this blog

We will learn the following things:

- Scatter plots and residual plots to visualise linear relationships
- Q-Q plots for normality
- Correlation matrices to determine co-linearity
- Residual plots for homoscedasticity

We will compare the expected plots (how the plots should look like if the assumptions are met) obtained from simulated data, with the plots obtained from a toy dataset from [Scikit-Learn](https://kgptalkie.com/feature-engineering-tutorial-series-4-linear-model-assumptions).

## Let’s Start!

We will start by importing all the libraries we need throughout this blog.

```python
import pandas as pd
import numpy as np
# for plotting and visualization
import matplotlib.pyplot as plt
import seaborn as sns
# for the Q-Q plots
import scipy.stats as stats
# the dataset for the demo
from sklearn.datasets import load_boston
# for linear regression
from sklearn.linear_model import LinearRegression
# to split and standardize the dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# to evaluate the regression model
from sklearn.metrics import mean_squared_error
```

We will load the Boston House price dataset from [sklearn](https://kgptalkie.com/feature-engineering-tutorial-series-4-linear-model-assumptions).

```python
boston_dataset = load_boston()
# create a dataframe with the independent variables
boston = pd.DataFrame(boston_dataset.data,
                      columns=boston_dataset.feature_names)
# add the target
boston['MEDV'] = boston_dataset.target
```

### Dataset Description

The Boston house-prices dataset has the following characteristics:

- **Number of Instances:** 506
- **Number of Attributes:** 13 numeric/categorical predictive
- **Target Variable:** Median Value (attribute 14)

**Source:** [https://kgptalkie.com/feature-engineering-tutorial-series-4-linear-model-assumptions](https://kgptalkie.com/feature-engineering-tutorial-series-4-linear-model-assumptions)

**Attribute Information:**

- **CRIM:** per capita crime rate by town
- **ZN:** proportion of residential land zoned for lots over 25,000 sq.ft.
- **INDUS:** proportion of non-retail business acres per town
- **CHAS:** Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
- **NOX:** nitric oxides concentration (parts per 10 million)
- **RM:** average number of rooms per dwelling
- **AGE:** proportion of owner-occupied units built prior to 1940
- **DIS:** weighted distances to five Boston employment centres
- **RAD:** index of accessibility to radial highways
- **TAX:** full-value property-tax rate per $10,000
- **PTRATIO:** pupil-teacher ratio by town
- **B:** 1000(Bk - 0.63)² where Bk is the proportion of blacks by town
- **LSTAT:** % lower status of the population
- **MEDV:** Median value of owner-occupied homes in $1000's

## Simulation Data

```python
np.random.seed(29)
n = 200
x = np.random.randn(n)
y = x * 10 + np.random.randn(n) * 2

toy_df = pd.DataFrame([x, y]).T
toy_df.columns = ['x', 'y']
```

### Linear Assumption

We evaluate linear assumption with scatter plots and residual plots. Scatter plots are used to plot the change in the dependent variable **y** with the independent variable **x**.

#### Scatter Plots

```python
sns.lmplot(x="x", y="y", data=toy_df, order=1)
plt.ylabel('Target')
plt.xlabel('Independent variable')
```

For the Boston house price dataset:

```python
sns.lmplot(x="LSTAT", y="MEDV", data=boston, order=1)
sns.lmplot(x="RM", y="MEDV", data=boston, order=1)
sns.lmplot(x="CRIM", y="MEDV", data=boston, order=1)
```

After log transformation of **CRIM**:

```python
boston['log_crim'] = np.log(boston['CRIM'])
sns.lmplot(x="log_crim", y="MEDV", data=boston, order=1)
boston.drop(labels='log_crim', inplace=True, axis=1)
```

### Assessing Linear Relationship by Examining Residuals

```python
linreg = LinearRegression()
linreg.fit(toy_df['x'].to_frame(), toy_df['y'])
pred = linreg.predict(toy_df['x'].to_frame())
error = toy_df['y'] - pred
plt.scatter(x=pred, y=toy_df['y'])
plt.xlabel('Predictions')
plt.ylabel('Real value')
```

Residuals plot:

```python
plt.scatter(y=error, x=toy_df['x'])
plt.ylabel('Residuals')
plt.xlabel('Independent variable x')
sns.distplot(error, bins=30)
plt.xlabel('Residuals')
```

### Multicollinearity

```python
correlation_matrix = boston[features].corr().round(2)
figure = plt.figure(figsize=(12, 12))
sns.heatmap(data=correlation_matrix, annot=True)
```

High correlation between **RAD** and **TAX** (0.91), **NOX** and **DIS** (-0.71).

### Normality

#### Histograms

```python
sns.distplot(toy_df['x'], bins=30)
sns.distplot(boston['RM'], bins=30)
sns.distplot(boston['LSTAT'], bins=30)
sns.distplot(np.log(boston['LSTAT']), bins=30)
```

#### Q-Q Plots

```python
stats.probplot(toy_df['x'], dist="norm", plot=plt)
plt.show()
stats.probplot(boston['RM'], dist="norm", plot=plt)
plt.show()
stats.probplot(boston['LSTAT'], dist="norm", plot=plt)
plt.show()
stats.probplot(np.log(boston['LSTAT']), dist="norm", plot=plt)
plt.show()
```

### Homoscedasticity

```python
X_train, X_test, y_train, y_test = train_test_split(
    boston[['RM', 'LSTAT', 'CRIM']],
    boston['MEDV'],
    test_size=0.3,
    random_state=0
)

scaler = StandardScaler()
scaler.fit(X_train)
linreg = LinearRegression()
linreg.fit(scaler.transform(X_train), y_train)
pred = linreg.predict(scaler.transform(X_test))
error = y_test - pred
```

Residuals plots:

```python
plt.scatter(x=X_test['LSTAT'], y=error)
plt.xlabel('LSTAT')
plt.ylabel('Residuals')
plt.scatter(x=X_test['RM'], y=error)
plt.xlabel('RM')
plt.ylabel('Residuals')
plt.scatter(x=X_test['CRIM'], y=error)
plt.xlabel('CRIM')
plt.ylabel('Residuals')
sns.distplot(error, bins=30)
```

After transformation:

```python
boston['LSTAT'] = np.log(boston['LSTAT'])
boston['CRIM'] = np.log(boston['CRIM'])
boston['RM'] = np.log(boston['RM'])
```

Residuals plots after transformation:

```python
plt.scatter(x=X_test['LSTAT'], y=error)
plt.xlabel('LSTAT')
plt.ylabel('Residuals')
plt.scatter(x=X_test['RM'], y=error)
plt.xlabel('RM')
plt.ylabel('Residuals')
plt.scatter(x=X_test['CRIM'], y=error)
plt.xlabel('CRIM')
plt.ylabel('Residuals')
sns.distplot(error, bins=30)
```

### Yellowbrick Residuals Plot

```python
from yellowbrick.regressor import ResidualsPlot
linreg = LinearRegression()
linreg.fit(scaler.transform(X_train), y_train)
visualizer = ResidualsPlot(linreg)
visualizer.fit(scaler.transform(X_train), y_train)
visualizer.score(scaler.transform(X_test), y_test)
visualizer.poof()
```

**Note:** Transforming the data improved the fit (Test R² of 0.65 for transformed data vs 0.6 for non-transformed data).