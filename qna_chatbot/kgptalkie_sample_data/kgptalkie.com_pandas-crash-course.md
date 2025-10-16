https://kgptalkie.com/pandas-crash-course

**Source:** https://kgptalkie.com/pandas-crash-course

# Pandas Crash Course

**Published by:** georgiannacambel  
**Date:** 17 September 2020

## What is Pandas?

**pandas** is a fast, powerful, flexible, and easy-to-use open-source data analysis and manipulation tool in **Python**. It is a high-level data manipulation tool developed by Wes McKinney. Built on the **Numpy** package, its key data structure is the **DataFrame**.

### Key Features:
- Stores and manipulates tabular data in rows of observations and columns of variables.
- Allows importing data from various formats (CSV, JSON, SQL, Excel).
- Supports merging, reshaping, selecting, data cleaning, and wrangling.
- Provides tools for data alignment, handling missing data, and more.

## DataFrame Object

A **DataFrame** is a two-dimensional, size-mutable, heterogeneous tabular data structure with labeled axes (rows and columns). It consists of:
- **Data**: The actual values.
- **Rows**: Observations.
- **Columns**: Variables.

### Example:
```python
import pandas as pd

data = {
    'apple': [3, 1, 4, 5],
    'orange': [1, 5, 6, 8]
}

df = pd.DataFrame(data)
print(df)
```

**Output:**
```
   apple  orange
0      3       1
1      1       5
2      4       6
3      5       8
```

## Reading CSV Files

Use `pd.read_csv()` to load data into a DataFrame.

```python
df = pd.read_csv('nba.csv')
print(df.head(10))
```

### Example Output:
```
        Name        Team  Number Position   Age  Height  Weight  College       Salary
0  Avery Bradley  Boston Celtics     0.0     PG  25.0    6-2   180.0      Texas  7730337.0
1    Jae Crowder  Boston Celtics    99.0     SF  25.0    6-6   235.0   Marquette  6796117.0
2    John Holland  Boston Celtics    30.0     SG  27.0    6-5   205.0  Boston University        NaN
```

## Handling Missing Data

### Detect Missing Values:
```python
print(df.isnull().sum())
```

**Output:**
```
Title            0
Genre            0
Description      0
Director         0
Actors           0
Year             0
Runtime          0
Rating           0
Votes            0
Revenue        128
Metascore       64
```

### Impute Missing Values:
```python
df['Revenue'].fillna(df['Revenue'].mean(), inplace=True)
df['Metascore'].fillna(df['Metascore'].mean(), inplace=True)
```

## Data Visualization

### Correlation Matrix:
```python
import seaborn as sns
import matplotlib.pyplot as plt

corrmat = df.corr()
sns.heatmap(corrmat)
plt.show()
```

### Scatter Plot:
```python
df.plot(kind='scatter', x='Rating', y='Revenue', title='Revenue vs Rating')
plt.show()
```

### Box Plot:
```python
df['Rating'].plot(kind='box', title='Rating Distribution')
plt.show()
```

## Adding New Columns

```python
rating_cat = []
for rate in df['Rating']:
    if rate > 6.2:
        rating_cat.append('Good')
    else:
        rating_cat.append('Bad')

df['Rating Category'] = rating_cat
print(df.head())
```

**Output:**
```
        Name        Team  Number Position   Age  Height  Weight  College       Salary  Rating  Rating Category
0  Avery Bradley  Boston Celtics     0.0     PG  25.0    6-2   180.0      Texas  7730337.0         8.1           Good
1    Jae Crowder  Boston Celtics    99.0     SF  25.0    6-6   235.0   Marquette  6796117.0         7.0           Good
2    John Holland  Boston Celtics    30.0     SG  27.0    6-5   205.0  Boston University        NaN         6.2           Bad
```

## Conclusion

This crash course covers the essentials of **pandas**, including data manipulation, handling missing values, and visualization. For more details, visit the [original source](https://kgptalkie.com/pandas-crash-course).

**Source:** https://kgptalkie.com/pandas-crash-course