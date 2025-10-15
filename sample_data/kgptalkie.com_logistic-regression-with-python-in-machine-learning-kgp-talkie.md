# Logistic Regression with Python in Machine Learning | KGP Talkie  
**Source:** [https://kgptalkie.com/logistic-regression-with-python-in-machine-learning-kgp-talkie](https://kgptalkie.com/logistic-regression-with-python-in-machine-learning-kgp-talkie)  

Published by  
KGP Talkie  
on  
8 August 2020  

---

## What is Logistic Regression?

**Logistic Regression** is a **Machine Learning** algorithm used for **classification problems**. It is a **predictive analysis algorithm** based on the concept of **probability**.  

**Logistic Regression** is a **supervised classification algorithm**. In a classification problem, the target variable (or output), `y`, can take only **discrete values** for a given set of features (or inputs), `X`.  

**Logistic Regression** becomes a classification technique only when a **decision threshold** is introduced. The setting of the threshold value is crucial and depends on the classification problem itself.  

### Examples of Classification Problems:
- Email: **Spam** or **Not Spam**  
- Online transactions: **Fraud** or **Not Fraud**  
- Tumor: **Malignant** or **Benign**  

**Logistic Regression** transforms its output using the **logistic sigmoid function** to return a **probability value**.  

We can call **Logistic Regression** a **Linear Regression model** but with a **more complex cost function**. This **cost function** is defined as the **Sigmoid function** (or **logistic function**) instead of a **linear function**.  

The **hypothesis** of **Logistic Regression** limits the **cost function** between **0** and **1**. Linear functions fail to represent this because they can produce values greater than **1** or less than **0**, which is not valid under the **Logistic Regression** hypothesis.  

---

## Types of Logistic Regression

Based on the number of categories, **Logistic Regression** can be classified as:

### 1. **Binomial**
- **Target variable** can have only **2** possible types: `"0"` or `"1"`.  
  - Examples: **Win vs Loss**, **Pass vs Fail**, **Dead vs Alive**.

### 2. **Multinomial**
- **Target variable** can have **3 or more** possible types (unordered).  
  - Examples: **Disease A vs Disease B vs Disease C**.

### 3. **Ordinal**
- **Target variable** has **ordered categories**.  
  - Example: **Test Score**: "Very Poor", "Poor", "Good", "Very Good" (mapped to scores: 0, 1, 2, 3).

---

## What is the Sigmoid Function?

To map predicted values to **probabilities**, we use the **Sigmoid function**. It maps any real value into a value between **0** and **1**.  

### Properties of the Sigmoid Function:
- **Maps predictions to probabilities**.
- **Output range**: `(0, 1)`.
- **Advantage**: Unlike linear functions, it avoids outputs outside this range.
- **Drawback**: **Vanishing gradients** near the ends of the curve (small gradients can hinder learning).

---

## Decision Boundary

We expect our classifier to return a **probability score** between **0** and **1** based on **probability** when inputs are passed through a prediction function.

---

## Cost Function

In **Linear Regression**, the **cost function** represents the **optimization objective**. However, using the **Linear Regression cost function** in **Logistic Regression** would result in a **non-convex function** with many local minima, making it difficult to find the **global minimum**.

### Logistic Regression Cost Function:
$$
\text{cost}(h(\theta), x) =
\begin{cases}
- \log(h_{\theta}(x)) & \text{if } y = 1 \\
- \log(1 - h_{\theta}(x)) & \text{if } y = 0
\end{cases}
$$

---

## Practical Example: Logistic Regression with the Titanic Dataset

### Problem Statement
Predict whether a passenger **survived** or **did not survive** the **RMS Titanic** disaster.

**Source:** [https://kgptalkie.com/logistic-regression-with-python-in-machine-learning-kgp-talkie](https://kgptalkie.com/logistic-regression-with-python-in-machine-learning-kgp-talkie)  

### Data Overview
The **RMS Titanic** sank on **15 April 1912** after colliding with an **iceberg** during its maiden voyage from **Southampton to New York City**. Approximately **2,224** passengers and crew were aboard, with over **1,500** deaths.

---

### Code Implementation

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.linear_model import LogisticRegression

# Load dataset
titanic = sns.load_dataset('titanic')
titanic.head(10)
```

### Data Preprocessing

```python
# Check for missing values
titanic.isnull().sum()

# Impute missing age values based on pclass
def impute_age(cols):
    age = cols[0]
    pclass = cols[1]
    if pd.isnull(age):
        if pclass == 1:
            return titanic[titanic['pclass'] == 1]['age'].mean()
        elif pclass == 2:
            return titanic[titanic['pclass'] == 2]['age'].mean()
        elif pclass == 3:
            return titanic[titanic['pclass'] == 3]['age'].mean()
    else:
        return age

titanic['age'] = titanic[['age', 'pclass']].apply(impute_age, axis=1)
```

### Convert Categorical Data to Numerical

```python
# Map categorical variables to numerical
genders = {'male': 0, 'female': 1}
titanic['sex'] = titanic['sex'].map(genders)

ports = {'S': 0, 'C': 1, 'Q': 2}
titanic['embarked'] = titanic['embarked'].map(ports)
```

### Build Logistic Regression Model

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split data
X = titanic.drop('survived', axis=1)
y = titanic['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train model
model = LogisticRegression(solver='lbfgs', max_iter=400)
model.fit(X_train, y_train)

# Evaluate model
y_predict = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_predict))
```

### Model Evaluation Metrics

```python
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc

# Predict probabilities
y_predict_prob = model.predict_proba(X_test)[:, 1]

# Compute ROC AUC
fpr, tpr, thr = roc_curve(y_test, y_predict_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='coral', label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
```

---

## Conclusion

**Logistic Regression** is a powerful tool for **binary classification** problems. By using the **Sigmoid function**, it maps predictions to **probabilities**, and with proper **feature engineering**, it can achieve high **accuracy** and **interpretability**.

**Source:** [https://kgptalkie.com/logistic-regression-with-python-in-machine-learning-kgp-talkie](https://kgptalkie.com/logistic-regression-with-python-in-machine-learning-kgp-talkie)