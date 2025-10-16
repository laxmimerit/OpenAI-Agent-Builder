# Linear Regression with Python | Machine learlearning | KGP Talkie  
**Source:** https://kgptalkie.com/linear-regression-with-python-machine-learlearning-kgp-talkie  

Published by  
**KGP Talkie**  
on  
**7 August 2020**  

---

## What is Linear Regression?

You are a **real estate** agent and you want to **predict** the house price. It would be great if you can make some kind of **automated system** which predicts price of a house based on various input which is known as **feature**.

**Supervised** Machine learning algorithms needs some data to train its **model** before making a prediction. For that we have a **Boston Dataset**.

---

## Where can Linear Regression be used?

It is a very **powerful technique** and can be used to understand the factors that influence **profitability**. It can be used to **forecast sales** in the coming months by analyzing the sales data for previous months. It can also be used to gain various insights about **customer behaviour**.

---

## What is Regression?

Let’s first understand what exactly **Regression** means. It is a statistical method used in finance, investing, and other disciplines that attempts to determine the **strength** and **character** of the relationship between one **dependent variable** (usually denoted by **Y**) and a series of other variables known as **independent variables**.

**Linear Regression** is a statistical technique where based on a set of **independent variable(s)** a dependent variable is **predicted**.

---

## Regression Examples

- **Stock prediction**: We can **predict** the price of **stock** depends on **dependent variable**, **x**. Let’s say recent history of stock price, news events.
- **Tweet popularity**: We can also **estimate** number of people will **retweet** for your tweet in **tewitter** based number of followers, popularity of hashtag.
- **In real estate**: As we discussed earlier, We can also **predict** the house prices and land prices in real estate.

---

## Regression Types

It is of two types:  
- **simple linear regression**  
- **multiple linear regression**

### Simple linear regression:
It is characterized by an **variable quantity**.

**Simple Linear Regression**  
$$ y_i = \beta_0 + \beta_1 X_i + \varepsilon_i $$  
- $ y $: dependent variable  
- $ \beta_0 $: population of intercept  
- $ \beta_i $: population of co-efficient  
- $ x $: independent variable  
- $ \varepsilon_i $: Random error  

### Multiple Linear Regression
It (as the name suggests) is characterized by **multiple independent variables** (more than **1**). While you discover the simplest **fit line**, you’ll be able to adjust a **polynomial or regression** toward the **mean**. And these are called **polynomial or regression** toward the **mean**.

---

## Assessing the performance of the model

How do we determine the best fit line?  
The line for which the **error** between the **predicted values** and the **observed values** is minimum is called the **best fit line** or **the regression line**. These errors are also called as **residuals**. The residuals can be visualized by the vertical lines from the observed data value to the **regression line**.

---

## Bias-Variance tradeoff

**Bias** are the simplifying **assumptions** made by a model to make the target function easier to learn. **Variance** is the amount that the estimate of the target function will change if different training data was used. The goal of any supervised **machine learning** algorithm is to achieve **low bias and low variance**. In turn the algorithm should achieve **good** prediction performance.

---

## How to determine error

### Gradient descent algorithm

**Gradient descent** is the **backbone** of an **machine learning** algorithm. To estimate the predicted value for, **Y** we will start with **random value** for, **θ** then derive cost using the above equation which stands for **Mean Squared Error(MSE)**. Remember we will try to get the **minimum value** of **cost function** that we will get by **derivation** of **cost function**.

---

## Gradient Descent Algorithm to reduce the cost function

You might not end up in global minimum.

---

## Implimentation with sklearn

### scikit-learn  
**Machine Learning in Python**  
- Simple and efficient tools for data mining and data analysis  
- Accessible to everybody, and reusable in various contexts  
- Built on NumPy, SciPy, and matplotlib  
- Open source, commercially usable – BSD license  

**Learn more here:**  
https://scikit-learn.org/stable/  

**Image Source:**  
https://cdn-images-1.medium.com/max/2400/1*2NR51X0FDjLB13u4WdYc4g.png  

---

## Training and testing splitting

Let’s discuss something about training a **ML model**, this model generally will try to **predict** one variable based on all the others. To verify how well this **model** works, we need a second data set, the **test set**. We use the model we learned from the **training data** and see how well it predicts the variable in question for the **training set**. When given a **data set** for which you want to use **Machine Learning**, typically you would divide it randomly into 2 sets. One will be used for **training**, the other for **testing**.

---

## Lets get started

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

boston = load_boston()
type(boston)
```

**Output:**  
`sklearn.utils.Bunch`

```python
boston.keys()
```

**Output:**  
`dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename'])`

```python
print(boston.DESCR)
```

---

## Boston house prices dataset

**Source:** https://kgptalkie.com/linear-regression-with-python-machine-learlearning-kgp-talkie  

**Source:** https://kgptalkie.com/linear-regression-with-python-machine-learlearning-kgp-talkie  

**Data Set Characteristics:**  
- **Number of Instances:** 506  
- **Number of Attributes:** 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.  

**Attribute Information (in order):**  
- CRIM: per capita crime rate by town  
- ZN: proportion of residential land zoned for lots over 25,000 sq.ft.  
- INDUS: proportion of non-retail business acres per town  
- CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)  
- NOX: nitric oxides concentration (parts per 10 million)  
- RM: average number of rooms per dwelling  
- AGE: proportion of owner-occupied units built prior to 1940  
- DIS: weighted distances to five Boston employment centres  
- RAD: index of accessibility to radial highways  
- TAX: full-value property-tax rate per $10,000  
- PTRATIO: pupil-teacher ratio by town  
- B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town  
- LSTAT: % lower status of the population  
- MEDV: Median value of owner-occupied homes in $1000's  

**Missing Attribute Values:** None  

**Creator:** Harrison, D. and Rubinfeld, D.L.  

**Source:** https://archive.ics.uci.edu/ml/machine-learning-databases/housing/  

---

## DataFrame()

A **Data frame** is a **two-dimensional** data structure, i.e., data is aligned in a **tabular fashion** in rows and columns.

```python
data = pd.DataFrame(data = data, columns= boston.feature_names)
data.head()
```

**Output:**  
| CRIM | ZN | INDUS | CHAS | NOX | RM | AGE | DIS | RAD | TAX | PTRATIO | B | LSTAT |  
|------|----|-------|------|-----|----|-----|-----|-----|-----|---------|---|-------|  
| 0.00632 | 18.0 | 2.31 | 0.0 | 0.538 | 6.575 | 65.2 | 4.0900 | 1.0 | 296.0 | 15.3 | 396.90 | 4.98 |  
| 0.02731 | 0.0 | 7.07 | 0.0 | 0.469 | 6.421 | 78.9 | 4.9671 | 2.0 | 242.0 | 17.8 | 396.90 | 9.14 |  

```python
data['Price'] = boston.target
data.head()
```

**Output:**  
| CRIM | ZN | INDUS | CHAS | NOX | RM | AGE | DIS | RAD | TAX | PTRATIO | B | LSTAT | Price |  
|------|----|-------|------|-----|----|-----|-----|-----|-----|---------|---|-------|-------|  
| 0.00632 | 18.0 | 2.31 | 0.0 | 0.538 | 6.575 | 65.2 | 4.0900 | 1.0 | 296.0 | 15.3 | 396.90 | 4.98 | 24.0 |  

---

## Understand your data

```python
data.describe()
```

**Output:**  
| CRIM | ZN | INDUS | CHAS | NOX | RM | AGE | DIS | RAD | TAX | PTRATIO | B | LSTAT | Price |  
|------|----|-------|------|-----|----|-----|-----|-----|-----|---------|---|-------|-------|  
| count | 506.000000 | 506.000000 | 506.000000 | 506.000000 | 506.000000 | 506.000000 | 506.000000 | 506.000000 | 506.000000 | 506.000000 | 506.000000 | 506.000000 | 506.000000 | 506.000000 |  
| mean | 3.613524 | 11.363636 | 11.136779 | 0.069170 | 0.554695 | 6.284634 | 68.574901 | 3.795043 | 9.549407 | 408.237154 | 18.455534 | 356.674032 | 12.653063 | 22.532806 |  
| std | 8.601545 | 23.322453 | 6.860353 | 0.253994 | 0.115878 | 0.702617 | 28.148861 | 2.105710 | 8.707259 | 168.537116 | 2.164946 | 91.294864 | 7.141062 | 9.197104 |  
| min | 0.006320 | 0.000000 | 0.460000 | 0.000000 | 0.385000 | 3.561000 | 2.900000 | 1.129600 | 1.000000 | 187.000000 | 12.600000 | 0.320000 | 1.730000 | 5.000000 |  
| 25% | 0.082045 | 0.000000 | 5.190000 | 0.000000 | 0.449000 | 5.885500 | 45.025000 | 2.100175 | 4.000000 | 279.000000 | 17.400000 | 375.377500 | 6.950000 | 17.025000 |  
| 50% | 0.256510 | 0.000000 | 9.690000 | 0.000000 | 0.538000 | 6.208500 | 77.500000 | 3.207450 | 5.000000 | 330.000000 | 19.050000 | 391.440000 | 11.360000 | 21.200000 |  
| 75% | 3.677083 | 12.500000 | 18.100000 | 0.000000 | 0.624000 | 6.623500 | 94.075000 | 5.188425 | 24.000000 | 666.000000 | 20.200000 | 396.225000 | 16.955000 | 25.000000 |  
| max | 88.976200 | 100.000000 | 27.740000 | 1.000000 | 0.871000 | 8.780000 | 100.000000 | 12.126500 | 24.000000 | 711.000000 | 22.000000 | 396.900000 | 37.970000 | 50.000000 |  

---

## Data Visualization

We will start by creating a **scatterplot matrix** that will allow us to visualize the **pair-wise relationships** and **correlations** between the different features. It is also quite useful to have a quick overview of how the data is distributed and whether it contains or not outliers.

```python
sns.pairplot(data)
plt.show()
```

```python
rows = 2
cols = 7
fig, ax = plt.subplots(nrows= rows, ncols= cols, figsize = (16,4))
col = data.columns
index = 0
for i in range(rows):
    for j in range(cols):
        sns.distplot(data[col[index]], ax = ax[i][j])
        index = index + 1
plt.tight_layout()
plt.show()
```

---

## Correlation Matrix

We are going to create now a **correlation matrix** to quantify and summarize the relationships between the variables. This **correlation matrix** is closely related with **covariance matrix**, in fact it is a rescaled version of the **covariance matrix**, computed from standardize features. It is a square matrix (with the same number of columns and rows) that contains the **Person’s r correlation coefficient**.

```python
corrmat = data.corr()
corrmat
```

**Output:**  
| CRIM | ZN | INDUS | CHAS | NOX | RM | AGE | DIS | RAD | TAX | PTRATIO | B | LSTAT | Price |  
|------|----|-------|------|-----|----|-----|-----|-----|-----|---------|---|-------|-------|  
| 1.000000 | -0.200469 | 0.406583 | -0.055892 | 0.420972 | -0.219247 | 0.352734 | -0.379670 | 0.625505 | 0.582764 | 0.289946 | -0.385064 | 0.455621 | -0.388305 |  
| -0.200469 | 1.000000 | -0.533828 | -0.042697 | -0.516604 | 0.311991 | -0.569537 | 0.664408 | -0.311948 | -0.314563 | -0.391679 | 0.175520 | -0.412995 | 0.360445 |  
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |  

---

## Heatmap

A **heatmap** is a **two-dimensional** graphical representation of data where the individual values that are contained in a **matrix** are represented as **colors**. The **seaborn** python package allows the creation of **annotated heatmaps** which can be tweaked using **Matplotlib** tools as per the creator’s requirement.

```python
fig, ax = plt.subplots(figsize = (18, 10))
sns.heatmap(corrmat, annot = True, annot_kws={'size': 12})
plt.show()
```

---

## Defining performance metrics

It is difficult to measure the **quality** of a given model without **quantifying** its performance over **training** and **testing**. This is typically done using some type of performance metric, whether it is through calculating some type of **error**, the goodness of fit, or some other useful measurement. For this project, you will be calculating the **coefficient of determination**, $ R^2 $, to quantify your model’s performance.

The **coefficient of determination** for a model is a useful statistic in **regression analysis**, as it often describes how “good” that model is at making predictions.

The values for $ R^2 $ range from **0** to **1**, which captures the percentage of **squared correlation** between the predicted and actual values of the target variable. A model with an $ R^2 $ of **0** always **fails** to predict the target variable, whereas a model with an $ R^2 $ of **1** perfectly **predicts** the target variable. Any value between **0** and **1** indicates what **percentage** of the target variable, using this model, can be explained by the features. A model can be given a negative $ R^2 $ as well, which indicates that the model is no better than one that naively predicts the **mean** of the target variable.

---

## Regression Evaluation Metrics

Here are three common evaluation metrics for regression problems:

- **Mean Absolute Error (MAE)** is the mean of the **absolute value of the errors**:  
  $$ \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i| $$

- **Mean Squared Error (MSE)** is the mean of the **squared errors**:  
  $$ \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_线性回归的均方误差 (MSE) 是误差平方的均值：  
  $$ \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 $$

- **Root Mean Squared Error (RMSE)** is the square root of the **mean of the squared errors**:  
  $$ \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2} $$

---

## Performance Metrics Calculation

```python
from sklearn.metrics import r2_score
score = r2_score(y_test, y_predict)
mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
print('r2_score: ', score)
print('mae: ', mae)
print('mse: ', mse)
```

**Output:**  
```
r2_score:  0.48816420156925056
mae:  4.404434993909258
mse:  41.67799012221684
```

---

## Plotting Learning Curves

```python
from sklearn.model_selection import learning_curve, ShuffleSplit

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    
    plt.legend(loc="best")
    return plt

X = correlated_data.drop(labels = ['Price'], axis = 1)
y = correlated_data['Price']

title = "Learning Curves (Linear Regression) " + str(X.columns.values)

cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = LinearRegression()
plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=-1)
plt.show()
```

**Source:** https://kgptalkie.com/linear-regression-with-python-machine-learlearning-kgp-talkie