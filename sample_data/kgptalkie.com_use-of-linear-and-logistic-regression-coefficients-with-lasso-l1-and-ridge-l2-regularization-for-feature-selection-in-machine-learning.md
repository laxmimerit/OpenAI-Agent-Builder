https://kgptalkie.com/use-of-linear-and-logistic-regression-coefficients-with-lasso-l1-and-ridge-l2-regularization-for-feature-selection-in-machine-learning

# Use of Linear and Logistic Regression Coefficients with Lasso (L1) and Ridge (L2) Regularization for Feature Selection in Machine Learning

**Source:** https://kgptalkie.com/use-of-linear-and-logistic-regression-coefficients-with-lasso-l1-and-ridge-l2-regularization-for-feature-selection-in-machine-learning

**Source:** https://kgptalkie.com/use-of-linear-and-logistic-regression-coefficients-with-lasso-l1-and-ridge-l2-regularization-for-feature-selection-in-machine-learning

**Source:** https://kgptalkie.com/use-of-linear-and-logistic-regression-coefficients-with-lasso-l1-and-ridge-l2-regularization-for-feature-selection-in-machine-learning

Published by  
KGP Talkie  
on  
10 August 2020  
10 August 2020  

Watch Full Playlist:  
https://www.youtube.com/playlist?list=PLc2rvfiptPSQYzmDIFuq2PqN2n28ZjxDH

## Linear Regression

Let’s first understand what exactly **linear regression** is. It is a straightforward approach to predict the response $ y $ based on different prediction variables such as $ x $ and $ \varepsilon $. There is a linear relationship between $ x $ and $ y $.

$$
y_i = \beta_0 + \beta_1 X_i + \varepsilon_i
$$

- $ y $: dependent variable  
- $ \beta_0 $: population of intercept  
- $ \beta_i $: population of coefficient  
- $ x $: independent variable  
- $ \varepsilon_i $: Random error  

### Basic Assumptions

- Linear relationship with the target $ y $  
- Feature space $ X $ should have Gaussian distribution  
- Features are not correlated with each other  
- Features are in the same scale (i.e., have the same variance)  

## Lasso (L1) and Ridge (L2) Regularization

Regularization is a technique to discourage the **complexity** of the model. It does this by penalizing the **loss function**. This helps to solve the **overfitting** problem.

### L1 Regularization (also called Lasso)

- **Shrinks** the coefficients which are less important to **zero**.  
- With **Lasso regularization**, we can **remove** some features.

### L2 Regularization (also called Ridge)

- Doesn’t **reduce** the coefficients to **zero** but reduces the **regression coefficients**.  
- With this reduction, we can identify which feature has more importance.

### L1/L2 Regularization (also called Elastic Net)

A regression model that uses L1 regularization is called **Lasso Regression**, and a model that uses L2 is called **Ridge Regression**.

## What is Lasso Regularization

### 3 Sources of Error

1. **Noise**: We can’t do anything with the noise.  
2. **Bias error**: Quantifies how much on average the predicted values are different from the actual value.  
3. **Variance**: Quantifies how predictions made on the same observation differ from each other.

### Bias-Variance Trade-off

By increasing **model complexity**, total error will **decrease** till some point and then start to **increase**. We need to select **optimum model complexity** to get less error.

- If you get **high bias**, increase **model complexity**.  
- If you get **high variance**, decrease **model complexity**.

### L1 Regularization

The L1 regularization adds a penalty equal to the **sum** of the absolute value of the **coefficients**.  

From the following figure, L1 regularization will **shrink** some parameters to **zero**. Hence, some variables will not play any role in the model to get the final output. **L1 regression** can be seen as a way to select features in a model.

### How to Choose $ \lambda $

Split data into three sets:

- **Training set**: Fit the model and set regression coefficients with regularization.  
- **Validation set**: Test the model’s performance to select $ \lambda $.  
- **Test set**: Generalize testing.

## What is Ridge Regularization

### Ridge Regularization

The **L2 regularization** adds a penalty equal to the sum of the **squared value of the coefficients**.

- $ \lambda $: **tuning parameter** or **optimization parameter**.  
- $ w $: **regression coefficient**.

- If $ \lambda $ is **high**, we get **high bias** and **low variance**.  
- If $ \lambda $ is **low**, we get **low bias** and **high variance**.

We find the **optimized value of $ \lambda $** by tuning parameters. $ \lambda $ is the **strength of the regularization**.

The L2 regularization will force the parameters to be **relatively small**. The bigger the penalization, the smaller (and the more robust) the coefficients are.

## Difference Between L1 and L2 Regularization

| **L1 Regularization** | **L2 Regularization** |
|----------------------|-----------------------|
| Penalizes sum of absolute value of weights | Penalizes sum of square weights |
| Has a sparse solution | Has a non-sparse solution |
| Has multiple solutions | Has one solution |
| Has built-in feature selection | Has no feature selection |
| Is robust to outliers | Is not robust to outliers |
| Generates simple and interpretable models | Gives better prediction when output is a function of all input features |

## Load the Dataset

### Loading Required Libraries

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_selection import SelectFromModel
```

### Load and Preprocess Data

```python
titanic = sns.load_dataset('titanic')
titanic.head()
```

| survived | pclass | sex   | age    | sibsp | parch | fare     | embarked | class  | who   | adult_male | deck | embark_town | alive | alone |
|----------|--------|-------|--------|-------|-------|----------|----------|--------|-------|------------|------|-------------|-------|-------|
| 0        | 3      | male  | 22.0   | 1     | 0     | 7.2500   | S        | Third  | man   | True       | NaN  | Southampton | no    | False |
| 1        | 1      | female| 38.0   | 1     | 0     | 71.2833  | C        | First  | woman | False      | C    | Cherbourg   | yes   | False |
| 1        | 3      | female| 26.0   | 0     | 0     | 7.9250   | S        | Third  | woman | False      | NaN  | Southampton | yes   | True  |
| 1        | 1      | female| 35.0   | 1     | 0     | 53.1000  | S        | First  | woman | False      | C    | Southampton | yes   | False |
| 0        | 3      | male  | 35.0   | 0     | 0     | 8.0500   | S        | Third  | man   | True       | NaN  | Southampton | no    | True  |

```python
titanic.isnull().sum()
```

```python
titanic.drop(labels=['age', 'deck'], axis=1, inplace=True)
titanic = titanic.dropna()
titanic.isnull().sum()
```

### Feature Selection

```python
data = titanic[['pclass', 'sex', 'sibsp', 'parch', 'embarked', 'who', 'alone']].copy()
data.head()
```

```python
data.isnull().sum()
```

```python
sex = {'male': 0, 'female': 1}
data['sex'] = data['sex'].map(sex)
ports = {'S': 0, 'C': 1, 'Q': 2}
data['embarked'] = data['embarked'].map(ports)
who = {'man': 0, 'woman': 1, 'child': 2}
data['who'] = data['who'].map(who)
alone = {True: 1, False: 0}
data['alone'] = data['alone'].map(alone)
data.head()
```

### Train-Test Split

```python
X = data.copy()
y = titanic['survived']
X.shape, y.shape
```

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)
```

## Estimation of Coefficients of Linear Regression

```python
sel = SelectFromModel(LinearRegression())
sel.fit(X_train, y_train)
```

```python
sel.get_support()
```

```python
sel.estimator_.coef_
```

```python
mean = np.mean(np.abs(sel.estimator_.coef_))
mean
```

```python
features = X_train.columns[sel.get_support()]
features
```

```python
X_train_reg = sel.transform(X_train)
X_test_reg = sel.transform(X_test)
X_test_reg.shape
```

### Random Forest Accuracy

```python
def run_randomForest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy: ', accuracy_score(y_test, y_pred))
```

```python
%%time
run_randomForest(X_train_reg, X_test_reg, y_train, y_test)
```

```python
%%time
run_randomForest(X_train, X_test, y_train, y_test)
```

```python
X_train.shape
```

## Logistic Regression Coefficient with L1 Regularization

```python
sel = SelectFromModel(LogisticRegression(penalty='l1', C=0.05, solver='liblinear'))
sel.fit(X_train, y_train)
```

```python
sel.get_support()
```

```python
sel.estimator_.coef_
```

```python
X_train_l1 = sel.transform(X_train)
X_test_l1 = sel.transform(X_test)
```

```python
%%time
run_randomForest(X_train_l1, X_test_l1, y_train, y_test)
```

## L2 Regularization

```python
sel = SelectFromModel(LogisticRegression(penalty='l2', C=0.05, solver='liblinear'))
sel.fit(X_train, y_train)
```

```python
sel.get_support()
```

```python
sel.estimator_.coef_
```

```python
X_train_l1 = sel.transform(X_train)
X_test_l1 = sel.transform(X_test)
```

```python
%%time
run_randomForest(X_train_l1, X_test_l1, y_train, y_test)
```