# Types of Data Every Data Scientist Should Know  
**Source:** https://kgptalkie.com/types-of-data-every-data-scientist-should-know  

---

## Published by  
**georgiannacambel**  
**26 September 2020**  

---

## Introduction  
One of the central concepts of data science is gaining insights from data. Statistics is an excellent tool for unlocking such insights in data. In this post, we’ll see some basic types of data (variable) which can be present in your dataset.  

---

## What is a Variable?  
A variable is any characteristic, number, or quantity that can be measured or counted. They are called ‘variables’ because the value they take may vary, and it usually does.  

### Examples of Variables:  
- **Age**: 21, 35, 62, …  
- **Gender**: male, female  
- **Income**: GBP 20000, GBP 35000, GBP 45000, …  
- **House price**: GBP 350000, GBP 570000, …  
- **Country of birth**: China, Russia, Costa Rica, …  
- **Eye colour**: brown, green, blue, …  
- **Vehicle make**: Ford, Volkswagen, …  

Most variables in a dataset can be classified into one of four major types:  
1. **Numerical variables**  
2. **Categorical variables**  
3. **Date-Time Variables**  
4. **Mixed Variables**  

---

## Numerical Variables  
The values of a numerical variable are numbers. They can be further classified into:  

### Discrete Variables  
- Values are whole numbers (counts).  
- **Examples**:  
  - Number of active bank accounts of a borrower (1, 4, 7, …)  
  - Number of pets in the family  
  - Number of children in the family  

### Continuous Variables  
- May contain any value within a range.  
- **Examples**:  
  - House price (GBP 350000, 57000, 100000, …)  
  - Time spent surfing a website (3.4 seconds, 5.10 seconds, …)  
  - Total debt as percentage of total income in the last month (0.2, 0.001, 0, 0.75, …)  

---

## Dataset  
In this demo, we will use a toy dataset from a peer-to-peer finance company:  

| Column Name              | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `customer_id`           | Unique ID for each customer                                                 |
| `disbursed_amount`      | Loan amount given to the borrower                                           |
| `interest`              | Interest rate                                                             |
| `income`                | Annual income                                                             |
| `number_open_accounts`  | Open accounts (more on this later)                                          |
| `number_credit_lines_12`| Accounts opened in the last 12 months                                       |
| `target`                | Loan status (paid or being repaid = 1, defaulted = 0)                       |
| `loan_purpose`          | Intended use of the loan                                                    |
| `market`                | Risk market assigned to the borrower (based on their financial situation)   |
| `householder`           | Whether the borrower owns or rents their property                           |
| `date_issued`           | Date the loan was issued                                                    |
| `date_last_payment`     | Date of last payment towards repaying the loan                              |

---

## Code: Importing Libraries  
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

---

## Continuous Variables  
### Example: `disbursed_amount`  
```python
data['disbursed_amount'].unique()
```
Output:  
```
array([23201.5 ,  7425.  , 11150.  , ...,  6279.  , 12894.75, 25584.  ])
```

### Visualization  
```python
fig = data['disbursed_amount'].hist(bins=50)
fig.set_title('Loan Amount Requested')
fig.set_xlabel('Loan Amount')
fig.set_ylabel('Number of Loans')
```

---

## Discrete Variables  
### Example: `number_open_accounts`  
```python
data['number_open_accounts'].unique()
```
Output:  
```
array([ 4., 13.,  8., 20., 14.,  5.,  9., 18., 16., 17., 12., 15.,  6.,
       10., 11.,  7., 21., 19., 26.,  2., 22., 27., 23., 25., 24., 28.,
        3., 30., 41., 32., 33., 31., 29., 37., 49., 34., 35., 38.,  1.,
       36., 42., 47., 40., 44., 43.])
```

### Visualization  
```python
fig = data['number_open_accounts'].hist(bins=100)
fig.set_xlim(0, 30)
fig.set_title('Number of open accounts')
fig.set_xlabel('Number of open accounts')
fig.set_ylabel('Number of Customers')
```

---

## Binary Variables  
### Example: `target`  
```python
data['target'].unique()
```
Output:  
```
array([0, 1], dtype=int64)
```

---

## Categorical Variables  
A categorical variable can take on one of a limited, fixed number of possible values.  

### Ordinal Variables  
- Categories can be meaningfully ordered.  
- **Examples**:  
  - Student’s grade in an exam (A, B, C, Fail)  
  - Days of the week (Monday = 1, Sunday = 7)  

### Nominal Variables  
- No intrinsic order.  
- **Examples**:  
  - Country of birth (Argentina, England, Germany)  
  - Car colour (blue, grey, silver)  

---

## Example: `householder`  
```python
data['householder'].unique()
```
Output:  
```
array(['RENT', 'OWNER', 'MORTGAGE'], dtype=object)
```

### Visualization  
```python
fig = data['householder'].value_counts().plot.bar()
fig.set_title('Householder')
fig.set_ylabel('Number of customers')
```

---

## Date-Time Variables  
### Example: `date_issued`  
```python
data['date_issued_dt'] = pd.to_datetime(data['date_issued'])
```

### Visualization  
```python
data['month'] = data['date_issued_dt'].dt.month
data['year'] = data['date_issued_dt'].dt.year

fig = data.groupby(['year','month', 'market'])['disbursed_amount'].sum().unstack().plot(
    figsize=(14, 8), linewidth=2)
fig.set_title('Disbursed amount in time')
fig.set_ylabel('Disbursed Amount')
```

---

## Mixed Variables  
Mixed variables contain both numbers and labels.  

### Example: `open_il_24m`  
```python
data.open_il_24m.unique()
```
Output:  
```
array(['C', 'A', 'B', '0.0', '1.0', '2.0', '4.0', '3.0', '6.0', '5.0',
       '9.0', '7.0', '8.0', '13.0', '10.0', '19.0', '11.0', '12.0',
       '14.0', '15.0'], dtype=object)
```

### Visualization  
```python
fig = data.open_il_24m.value_counts().plot.bar()
fig.set_title('Number of installment accounts open')
fig.set_ylabel('Number of borrowers')
```

---

## Summary  
- **Numerical variables** include **discrete** and **continuous** types.  
- **Categorical variables** are further divided into **ordinal** and **nominal**.  
- **Date-time variables** require special handling for analysis.  
- **Mixed variables** combine numerical and categorical data.  

**Source:** https://kgptalkie.com/types-of-data-every-data-scientist-should-know  

---  
**Source:** https://kgptalkie.com/types-of-data-every-data-scientist-should-know