https://kgptalkie.com/data-visualization-with-pandas

# Data Visualization with Pandas

**Source:** https://kgptalkie.com/data-visualization-with-pandas

**Source:** https://kgptalkie.com/data-visualization-with-pandas

**Source:** https://kgptalkie.com/data-visualization-with-pandas

Published by  
georgiannacambel  
on  
18 September 2020

## Introduction

Data visualization is the discipline of trying to understand data by placing it in a visual context so that patterns, trends and correlations that might not otherwise be detected can be exposed.

`pandas` is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, in Python programming language. It is a high-level data manipulation tool developed by Wes McKinney. It is built on the Numpy package and its key data structure is called the DataFrame. DataFrames allow you to store and manipulate tabular data in rows of observations and columns of variables. Pandas is mainly used for data analysis. Pandas allows importing data from various file formats such as comma-separated values, JSON, SQL, Microsoft Excel. Pandas allows various data manipulation operations such as merging, reshaping, selecting, as well as data cleaning, and data wrangling features. Doing visualizations with pandas comes in handy when you want to view how your data looks like quickly.

## Setting Up

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from numpy.random import randn, randint, uniform, sample
```

## Pandas DataFrame

A DataFrame is two-dimensional size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns). A Data frame is a two-dimensional data structure, i.e., data is aligned in a tabular fashion in rows and columns. Pandas DataFrame consists of three principal components, the data, rows, and columns.

`Series` is a one-dimensional labeled array capable of holding data of any type (integer, string, float, python objects, etc.). Each column of the dataframe is a Series.

### Example: Creating a DataFrame

```python
df = pd.DataFrame(randn(1000), index = pd.date_range('2019-06-07', periods = 1000), columns=['value'])
ts = pd.Series(randn(1000), index = pd.date_range('2019-06-07', periods = 1000))
```

### Output

```python
df.head()
value
2019-06-07 -0.992350
2019-06-08 -0.849183
2019-06-09  0.126559
2019-06-10  0.640230
2019-06-11 -0.975090

ts.head()
2019-06-07    0.430385
2019-06-08    1.810955
2019-06-09    3.207345
2019-06-10   -0.366252
2019-06-11    1.406304
Freq: D, dtype: float64
```

```python
type(df), type(ts)
(pandas.core.frame.DataFrame, pandas.core.series.Series)
```

## Line Plot

The `cumsum()` function is used to get cumulative sum over a DataFrame or Series axis. It returns a DataFrame or Series of the same size containing the cumulative sum.

```python
df['value'] = df['value'].cumsum()
ts = ts.cumsum()
```

### Output

```python
df.head()
value
2019-06-07 -0.992350
2019-06-08 -2.833884
2019-06-09 -4.548859
2019-06-10 -5.623604
2019-06-11 -7.673438

ts.head()
2019-06-07     0.430385
2019-06-08     2.671725
2019-06-09     8.120411
2019-06-10    13.202844
2019-06-11    19.691581
Freq: D, dtype: float64
```

```python
type(df), type(ts)
(pandas.core.frame.DataFrame, pandas.core.series.Series)
```

### Visualization

```python
ts.plot(figsize=(10,5))
df.plot()
```

## Iris Dataset Example

```python
iris = sns.load_dataset('iris')
iris.head()
```

### Output

```python
sepal_length  sepal_width  petal_length  petal_width species
0           5.1          3.5           1.4          0.2  setosa
1           4.9          3.0           1.4          0.2  setosa
2           4.7          3.2           1.3          0.2  setosa
3           4.6          3.1           1.5          0.2  setosa
4           5.0          3.6           1.4          0.2  setosa
```

### Plotting

```python
ax = iris.plot(figsize=(15,8), title='Iris Dataset')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
```

```python
ts.plot(style = 'r--', label = 'Series', legend = True)
iris.plot(legend = False, figsize = (10, 5), logy = True)
```

## Secondary Axis Plot

```python
x = iris.drop(['sepal_width', 'petal_width'], axis = 1)
y = iris.drop(['sepal_length', 'petal_length'], axis = 1)
```

### Output

```python
x.head()
sepal_length  petal_length species
0           5.1           1.4   setosa
1           4.9           1.4   setosa
2           4.7           1.3   setosa
3           4.6           1.5   setosa
4           5.0           1.4   setosa

y.head()
sepal_width  petal_width species
0          3.5          0.2   setosa
1          3.0          0.2   setosa
2          3.2          0.2   setosa
3          3.1          0.2   setosa
4          3.6          0.2   setosa
```

### Visualization

```python
ax = x.plot()
y.plot(figsize = (16,10), secondary_y = True, ax = ax)
x.plot(figsize=(10,5), x_compat = True)
```

## Bar Plot

```python
df = iris.drop(['species'], axis = 1)
df.iloc[0].plot(kind='bar')
df.iloc[0].plot.bar()
```

## Titanic Dataset Example

```python
titanic = sns.load_dataset('titanic')
titanic.head()
```

### Output

```python
survived  pclass     sex   age  sibsp  parch     fare embarked  class    who  adult_male deck embark_town alive  alone
0       0       3    male  22.0      1      0   7.2500        S  Third     man        True  NaN   Southampton   no  False
1       1       1  female  38.0      1      0  71.2833        C  First   woman       False    C     Cherbourg   yes  False
2       1       3  female  26.0      0      0   7.9250        S  Third   woman       False  NaN   Southampton   yes   True
3       1       1  female  35.0      1      0  53.1000        S  First   woman       False    C   Southampton   yes  False
4       0       3    male  35.0      0      0   8.0500        S  Third     man        True  NaN   Southampton   no   True
```

### Histogram

```python
titanic['pclass'].plot(kind = 'hist')
```

## Random DataFrame Example

```python
df = pd.DataFrame(randn(10, 4), columns=['a', 'b', 'c', 'd'])
df.head()
```

### Output

```python
a         b         c         d
0 -0.358585 -0.530212 -1.037960 -0.620583
1  0.063102  0.872088  0.429474  2.020268
2 -1.064892 -0.521098 -0.238016  1.559072
3 -0.277393 -1.246629  1.723683 -0.069810
4 -1.123548 -0.375084  0.528301  0.739006
```

### Bar Plot

```python
df.plot.bar()
df.plot.bar(stacked = True)
df.plot.barh(stacked = True)
plt.axis('off')
```

## Histogram

```python
iris.plot.hist()
iris.plot(kind = 'hist')
iris.plot(kind = 'hist', stacked = True, bins = 50)
iris.plot(kind = 'hist', stacked = True, bins = 50, orientation = 'horizontal')
```

## Difference Plot

```python
iris['sepal_width'].diff()[:10]
iris['sepal_width'].diff().plot(kind = 'hist', stacked = True, bins = 50)
```

## Box Plot

```python
color = {'boxes': 'DarkGreen', 'whiskers': 'r'}
df.plot(kind = 'box', figsize=(10,5), color = color)
df.plot(kind = 'box', figsize=(10,5), color = color, vert = False)
```

## Area and Scatter Plot

```python
df.plot(kind = 'area')
df.plot.area(stacked = False)
df.plot.scatter(x = 'sepal_length', y = 'petal_length')
df.plot.scatter(x = 'sepal_length', y = 'petal_length', c = 'sepal_width')
```

### Grouped Scatter Plot

```python
ax = df.plot.scatter(x = 'sepal_length', y = 'petal_length', label = 'Length');
df.plot.scatter(x = 'sepal_width', y = 'petal_width', label = 'Width', ax = ax, color = 'r')
df.plot.scatter(x = 'sepal_length', y = 'petal_length', c = 'sepal_width', s = df['petal_width']*200)
```

## Hex and Pie Plot

```python
df.plot.hexbin(x = 'sepal_length', y = 'petal_length', gridsize = 10, C = 'sepal_width')
d = df.iloc[0]
d.plot.pie(figsize = (10,10))
```

### Multiple Pie Plots

```python
d = df.head(3).T
d.plot.pie(subplots = True, figsize = (20, 20))
d.plot.pie(subplots = True, figsize = (20, 20), fontsize = 16, autopct = '%.2f')
```

### Incomplete Pie Plot

```python
x=[0.2]*4
print(x)
print(sum(x))
series = pd.Series(x, index = ['a','b','c', 'd'], name = 'Pie Plot')
series.plot.pie()
```

## Scatter Matrix

```python
from pandas.plotting import scatter_matrix
scatter_matrix(df, figsize= (8,8), diagonal='kde', color = 'r')
plt.show()
```

## KDE Plots

```python
ts.plot.kde()
```

## Andrews Curves

```python
from pandas.plotting import andrews_curves
andrews_curves(df, 'sepal_width')
```

## Subplots

```python
df.plot(subplots = True, sharex = False)
plt.tight_layout()
df.plot(subplots = True, sharex = False, layout = (2,2), figsize = (16,8))
plt.tight_layout()
```

## Conclusion

Data visualization provides us with a quick, clear understanding of the information. Due to graphical representations, we can visualize large volumes of data in an understandable and coherent way, which in turn helps us comprehend the information and draw conclusions and insights. Relevant data visualization is essential for pinpointing the right direction to take for selecting and tuning a [machine learning](https://kgptalkie.com/data-visualization-with-pandas) model. It both shortens the machine learning process and provides more accuracy for its outcome.