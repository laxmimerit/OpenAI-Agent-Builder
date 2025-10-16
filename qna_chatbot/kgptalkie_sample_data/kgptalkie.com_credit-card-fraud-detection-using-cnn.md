# Credit Card Fraud Detection using CNN  
**Source:** [https://kgptalkie.com/credit-card-fraud-detection-using-cnn](https://kgptalkie.com/credit-card-fraud-detection-using-cnn)  

Published by [georgiannacambel](https://kgptalkie.com/credit-card-fraud-detection-using-cnn) on **3 September 2020**  

---

## Classification using CNN  
It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase. In this project, we are going to build a model using **CNN** which predicts if the transaction is genuine or fraudulent.  

---

## Dataset  
We are going to use the **[Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)** dataset from Kaggle. It contains anonymized credit card transactions labeled as fraudulent or genuine. You can download it from [here](https://www.kaggle.com/mlg-ulb/creditcardfraud).  

### Dataset Details  
- Contains transactions made by credit cards in September 2013 by European cardholders.  
- 492 frauds out of 284,807 transactions (highly unbalanced dataset).  
- **0.172%** of transactions are fraudulent.  
- Features:  
  - **'Time'**: Seconds elapsed between each transaction and the first transaction.  
  - **'Amount'**: Transaction amount (used for cost-sensitive learning).  
  - **'Class'**: Response variable (1 = fraud, 0 = genuine).  
  - Other features are PCA-transformed.  

---

## TensorFlow Installation  
We are going to use **[TensorFlow](https://www.tensorflow.org/)** to build the model. Install it using:  

```bash
!pip install tensorflow
!pip install tensorflow-gpu  # For GPU support
```

### Required Libraries  
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv1D, MaxPool1D
from tensorflow.keras.optimizers import Adam

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

---

## Data Preprocessing  
### Load Dataset  
```python
data = pd.read_csv('creditcard.csv')
data.head()
```

| Time | V1 | V2 | ... | V28 | Amount | Class |
|------|----|----|-----|-----|--------|-------|
| 0.0  | -1.359807 | -0.072781 | ... | -0.021053 | 149.62 | 0     |
| 0.0  | 1.191857 | 0.266151 | ... | 0.014724 | 2.69   | 0     |
| ...  | ...      | ...      | ... | ...   | ...    | ...   |

### Dataset Shape  
```python
data.shape  # (284807, 31)
```

### Check for Missing Values  
```python
data.isnull().sum()  # All values are 0 (no missing data)
```

### Class Distribution  
```python
data['Class'].value_counts()
```

```
0    284315
1       492
Name: Class, dtype: int64
```

---

## Balance Dataset  
```python
non_fraud = data[data['Class'] == 0]
fraud = data[data['Class'] == 1]

non_fraud = non_fraud.sample(fraud.shape[0])  # Balance classes
data = fraud.append(non_fraud, ignore_index=True)
```

### New Class Distribution  
```python
data['Class'].value_counts()
```

```
1    492
0    492
Name: Class, dtype: int64
```

---

## Train-Test Split  
```python
X = data.drop('Class', axis=1)
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
```

### Scale Features  
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
```

### Reshape for CNN  
```python
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
```

---

## Build CNN Model  
```python
model = Sequential()
model.add(Conv1D(32, 2, activation='relu', input_shape=X_train[0].shape))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv1D(64, 2, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))
```

### Model Summary  
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv1d (Conv1D)              (None, 29, 32)            96        
_________________________________________________________________
batch_normalization (BatchNo (None, 29, 32)            128       
_________________________________________________________________
dropout (Dropout)            (None, 29, 32)            0         
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 28, 64)            4160      
_________________________________________________________________
batch_normalization_1 (Batch (None, 28, 64)            256       
_________________________________________________________________
dropout_1 (Dropout)          (None, 28, 64)            0         
_________________________________________________________________
flatten (Flatten)            (None, 1792)              0         
_________________________________________________________________
dense (Dense)                (None, 64)                114752    
_________________________________________________________________
dropout_2 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 65        
=================================================================
Total params: 119,457
Trainable params: 119,265
Non-trainable params: 192
```

---

## Train Model  
```python
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=1)
```

### Training Results  
```
Epoch 20/20
787/787 [==============================] - 0s 399us/sample - loss: 0.2067 - accuracy: 0.9199 - val_loss: 0.2183 - val_accuracy: 0.8934
```

---

## Visualize Learning Curve  
```python
def plot_learningCurve(history, epoch):
    epoch_range = range(1, epoch+1)
    plt.plot(epoch_range, history.history['accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

plot_learningCurve(history, 20)
```

---

## Improve Model with MaxPooling  
```python
model = Sequential()
model.add(Conv1D(32, 2, activation='relu', input_shape=X_train[0].shape))
model.add(BatchNormalization())
model.add(MaxPool1D(2))
model.add(Dropout(0.2))

model.add(Conv1D(64, 2, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool1D(2))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=1)
```

### Training Results (After Improvements)  
```
Epoch 50/50
787/787 [==============================] - 0s 194us/sample - loss: 0.2445 - accuracy: 0.9123 - val_loss: 0.2449 - val_accuracy: 0.9137
```

---

**Source:** [https://kgptalkie.com/credit-card-fraud-detection-using-cnn](https://kgptalkie.com/credit-card-fraud-detection-using-cnn)