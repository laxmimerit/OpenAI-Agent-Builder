# Feature Engineering Series Tutorial 3: Rare Labels  
**Source:** [https://kgptalkie.com/feature-engineering-series-tutorial-3-rare-labels](https://kgptalkie.com/feature-engineering-series-tutorial-3-rare-labels)

## Published by  
georgiannacambel  
on  
1 October 2020  
1 October 2020  

---

### Labels that occur rarely  
Categorical variables are those whose values are selected from a group of categories, also called labels. Different labels appear in the dataset with different frequencies. Some categories appear more frequently in the dataset, whereas some other categories appear only in a few number of observations.

For example, in a dataset with information about loan applicants where one of the variables is “city” in which the applicant lives, cities like ‘New York’ may appear a lot in the data because New York has a huge population, whereas smaller towns like ‘Leavenworth’ will appear only on a few occasions (population < 2000 people), because the population there is very small. A borrower is more likely to live in New York, because far more people live in New York.

In fact, categorical variables often contain a few dominant labels that account for the majority of the observations and a large number of labels that appear only seldom.

---

### Are Rare Labels in a categorical variable a problem?  
Rare values can add a lot of information or none at all. For example, consider a stockholder meeting where each person can vote in proportion to their number of shares. One of the shareholders owns 50% of the stock, and the other 999 shareholders own the remaining 50%. The outcome of the vote is largely influenced by the shareholder who holds the majority of the stock. The remaining shareholders may have an impact collectively, but they have almost no impact individually.

The same occurs in real life datasets. The label that is over-represented in the dataset tends to dominate the outcome, and those that are under-represented may have no impact individually, but could have an impact if considered collectively.

More specifically:  
- Rare values in categorical variables tend to cause over-fitting, particularly in tree based methods.  
- A big number of infrequent labels adds noise, with little information, therefore causing over-fitting.  
- Rare labels may be present in training set, but not in test set, therefore causing over-fitting to the training set.  
- Rare labels may appear in the test set, and not in the train set. Thus, the **machine learning** model will not know how to evaluate it.

> **Note:** Sometimes rare values, are indeed important. For example, if we are building a model to predict fraudulent loan applications, which are by nature rare, then a rare value in a certain variable, may be indeed very predictive. This rare value could be telling us that the observation is most likely a fraudulent application, and therefore we would choose not to ignore it.

---

## In this Blog:  
We will:  
- Learn to identify rare labels in a dataset  
- Understand how difficult it is to derive reliable information from them.  
- Visualise the uneven distribution of rare labels between training and test sets.  

---

### Let’s start!  
Here we have imported the necessary libraries.  
- `pandas` is used to read the dataset into a dataframe and perform operations on it  
- `numpy` is used to perform basic array operations  
- `pyplot` from `matplotlib` is used to visualize the data  
- `train_test_split` is used to split the data into training and testing datasets.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
```

Now we will load the dataset with only the variables we need for this demo. We will read the dataset using `read_csv()` and display the first 5 rows using `head()`.

**Variable definitions:**  
- **Neighborhood:** Physical locations within Ames city limits.  
- **Exterior1st:** Exterior covering on house.  
- **Exterior2nd:** Exterior covering on house. (if more than one material)  
- **SalePrice:** The price of the house.

```python
use_cols = ['Neighborhood', 'Exterior1st', 'Exterior2nd', 'SalePrice']
data = pd.read_csv('https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/houseprice.csv', usecols=use_cols)
data.head()
```

---

### Cardinality of Categorical Variables  
Now we will look at the different number of unique labels in each variable i.e. we will find the cardinality of all the categorical variables.

```python
cat_cols = ['Neighborhood', 'Exterior1st', 'Exterior2nd']
for col in cat_cols:
    print('variable: ', col, ' number of labels: ', data[col].nunique())
```

**Output:**  
```
variable:  Neighborhood  number of labels:  25
variable:  Exterior1st  number of labels:  15
variable:  Exterior2nd  number of labels:  16
total houses:  1460
```

---

### Frequency Distribution of Labels  
Now we will plot the frequency of each label in the dataset for each variable. In other words, we will see the percentage of houses in the data with each label.

```python
total_houses = len(data)
for col in cat_cols:
    temp_df = pd.Series(data[col].value_counts() / total_houses)
    fig = temp_df.sort_values(ascending=False).plot.bar()
    fig.set_xlabel(col)
    fig.axhline(y=0.05, color='red')
    fig.set_ylabel('Percentage of houses')
    plt.show()
```

---

### Relationship Between Labels and Target Variable  
How is the target, `SalePrice`, related to these categories?

```python
def calculate_mean_target_per_category(df, var):
    total_houses = len(df)
    temp_df = pd.Series(df[var].value_counts() / total_houses).reset_index()
    temp_df.columns = [var, 'perc_houses']
    temp_df = temp_df.merge(df.groupby([var])['SalePrice'].mean().reset_index(), on=var, how='left')
    return temp_df
```

Now we will use the function to calculate the percentage of houses in each category of the variable `Neighborhood` and the mean `SalePrice` for each category.

```python
temp_df = calculate_mean_target_per_category(data, 'Neighborhood')
temp_df.head()
```

**Output:**  
| Neighborhood | perc_houses | SalePrice       |
|--------------|-------------|-----------------|
| NAmes        | 0.154110    | 145847.080000   |
| CollgCr      | 0.102740    | 197965.773333   |
| OldTown      | 0.077397    | 128225.300885   |
| Edwards      | 0.068493    | 128219.700000   |
| Somerst      | 0.058904    | 225379.837209   |

---

### Plotting Category Frequency and Mean SalePrice  
We will create a function to plot the category frequency and the mean `SalePrice`.

```python
def plot_categories(df, var):
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.xticks(df.index, df[var], rotation=90)
    ax2 = ax.twinx()
    ax.bar(df.index, df["perc_houses"], color='lightgrey')
    ax2.plot(df.index, df["SalePrice"], color='green', label='Seconds')
    ax.axhline(y=0.05, color='red')
    ax.set_ylabel('percentage of houses per category')
    ax.set_xlabel(var)
    ax2.set_ylabel('Average Sale Price per category')
    plt.show()
```

We will call the function for the variable `Neighborhood`.

```python
plot_categories(temp_df, 'Neighborhood')
```

---

### Grouping Rare Labels  
One common way of working with rare or infrequent values is to group them under an umbrella category called ‘Rare’ or ‘Other’.

```python
def group_rare_labels(df, var):
    total_houses = len(df)
    temp_df = pd.Series(df[var].value_counts() / total_houses)
    grouping_dict = {
        k: ('rare' if k not in temp_df[temp_df >= 0.05].index else k)
        for k in temp_df.index
    }
    tmp = df[var].map(grouping_dict)
    return tmp
```

We will start by grouping the rare labels in `Neighborhood`.

```python
data['Neighborhood_grouped'] = group_rare_labels(data, 'Neighborhood')
data[['Neighborhood', 'Neighborhood_grouped']].head(10)
```

**Output:**  
| Neighborhood | Neighborhood_grouped |
|--------------|----------------------|
| CollgCr      | CollgCr              |
| Veenker      | rare                 |
| CollgCr      | CollgCr              |
| Crawfor      | rare                 |
| NoRidge      | rare                 |
| Mitchel      | rare                 |
| Somerst      | Somerst              |
| NWAmes       | NWAmes               |
| OldTown      | OldTown              |
| BrkSide      | rare                 |

---

### Rare Labels in Training and Test Sets  
Now we will split the data into training and testing set.

```python
X_train, X_test, y_train, y_test = train_test_split(data[cat_cols],
                                                    data['SalePrice'],
                                                    test_size=0.3,
                                                    random_state=2910)
```

**Output:**  
```
X_train.shape, X_test.shape
((1022, 3), (438, 3))
```

Now let’s find the labels of variable `Exterior1st` that are present only in the training set.

```python
unique_to_train_set = [x for x in X_train['Exterior1st'].unique() if x not in X_test['Exterior1st'].unique()]
print(unique_to_train_set)
```

**Output:**  
```
['Stone', 'BrkComm', 'ImStucc', 'CBlock']
```

Now let’s find the labels present only in the test set.

```python
unique_to_test_set = [x for x in X_test['Exterior1st'].unique() if x not in X_train['Exterior1st'].unique()]
print(unique_to_test_set)
```

**Output:**  
```
['AsphShn']
```

---

**Source:** [https://kgptalkie.com/feature-engineering-series-tutorial-3-rare-labels](https://kgptalkie.com/feature-engineering-series-tutorial-3-rare-labels)