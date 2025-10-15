# Missing Values and its Mechanisms  
**Source:** https://kgptalkie.com/missing-values-and-its-mechanisms  

**Source:** https://kgptalkie.com/missing-values-and-its-mechanisms  

**Source:** https://kgptalkie.com/missing-values-and-its-mechanisms  

## Feature Engineering Series Tutorial 1: Missing Values and its Mechanisms  
**Published by:** georgiannacambel  
**Date:** 28 September 2020  

### What Are Missing Values?  
Missing data, or missing values, occur when **no data** / **no value** is stored for certain observations within a variable. Incomplete data is an unavoidable problem in most data sources and may significantly impact the conclusions derived from the data.  

### Why Is Data Missing?  
The source of missing data can vary. Examples include:  
- A value is missing due to being forgotten, lost, or not stored properly  
- For certain observations, the value does not exist  
- The value cannot be known or identified  
- Incomplete form submissions by users or data entry personnel  

### Importance of Understanding Missing Data Mechanisms  
It is critical to understand **how missing data is introduced** into a dataset. The **mechanisms** by which missing information is introduced determine how we process the missing values. Knowing the source of missing data may also help mitigate future data collection issues.  

---

## Missing Data Mechanisms  

### 1. Missing Completely at Random (MCAR)  
A variable is **MCAR** if the probability of being missing is the same for all observations. There is **no relationship** between missing data and any other values (observed or missing) in the dataset.  

**Example:**  
If values are missing completely at random, disregarding those cases would not bias inferences.  

### 2. Missing at Random (MAR)  
**MAR** occurs when the probability of missing values depends on **observed data**. For example, if men are more likely to disclose their weight than women, weight is MAR.  

**Example:**  
Including gender in analysis can help control bias in weight for missing observations.  

### 3. Missing Not at Random (MNAR)  
**MNAR** occurs when missing data is systematically related to the outcome. For example:  
- People failing to fill a depression survey due to their depression level  
- Fraudsters not uploading documents to avoid detection  

**Example:**  
There is a systematic relationship between missing documents and the target variable (fraud).  

---

## Detecting and Quantifying Missing Values  

### Example: Titanic Dataset  
```python
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
data = pd.read_csv("https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/titanic.csv")
data.head()
```

**Output:**  
```
   PassengerId  Survived  Pclass  ...  Cabin Embarked
0            1         0       3  ...   NaN        S
1            2         1       1  ...  C85        C
2            3         1       3  ...   NaN        S
3            4         1       1  ...  C123        S
4            5         0       3  ...   NaN        S
```

### Counting Missing Values  
```python
data.isnull().sum()
```

**Output:**  
```
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
```

### Percentage of Missing Values  
```python
data.isnull().mean()
```

**Output:**  
```
PassengerId    0.000000
Survived       0.000000
Pclass         0.000000
Name           0.000000
Sex            0.000000
Age            0.198653
Cabin          0.771044
Embarked       0.002245
dtype: float64
```

---

## Mechanisms of Missing Data  

### MNAR: Systematic Missing Values (Titanic Dataset)  
**Example:**  
- Missing values in `Age` and `Cabin` are systematically related to survival.  

```python
data['cabin_null'] = np.where(data['Cabin'].isnull(), 1, 0)
data.groupby(['Survived'])['cabin_null'].mean()
```

**Output:**  
```
Survived
0    0.876138
1    0.602339
Name: cabin_null, dtype: float64
```

**Observation:**  
- 87% of missing `Cabin` values are for non-survivors vs. 60% for survivors.  

### MCAR: Embarked Variable  
```python
data[data['Embarked'].isnull()]
```

**Output:**  
```
   PassengerId  Survived  Pclass  ...  Cabin Embarked
61          62         1       1  ...   B28      NaN
829        830         1       1  ...   B28      NaN
```

**Observation:**  
- No apparent relationship between missing `Embarked` values and other variables.  

### MAR: Loan Dataset Example  
```python
data = pd.read_csv('https://raw.githubusercontent.com/laxmimerit/feature-engineering-for-machine-learning-dataset/master/loan.csv', usecols=['employment', 'time_employed'])
data.head()
```

**Output:**  
```
  employment time_employed
0     Teacher     <=5 years
1   Accountant     <=5 years
2  Statistician     <=5 years
3        Other     <=5 years
4    Bus driver     >5 years
```

**Missing Data Analysis:**  
```python
data.isnull().mean()
```

**Output:**  
```
employment       0.0611
time_employed    0.0529
dtype: float64
```

**Observation:**  
- Missing values in `employment` are related to missing values in `time_employed`.  

---

## Conclusion  
Understanding the **mechanism** of missing data is crucial for choosing appropriate imputation strategies.  
- **MCAR:** Randomly missing data  
- **MAR:** Missing values depend on observed data  
- **MNAR:** Systematic missing values related to the outcome  

**Source:** https://kgptalkie.com/missing-values-and-its-mechanisms