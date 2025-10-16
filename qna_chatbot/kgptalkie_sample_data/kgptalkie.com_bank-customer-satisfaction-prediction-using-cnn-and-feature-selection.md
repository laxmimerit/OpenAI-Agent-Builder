# Bank Customer Satisfaction Prediction Using CNN and Feature Selection  
**Source:** https://kgptalkie.com/bank-customer-satisfaction-prediction-using-cnn-and-feature-selection  

Published by [georgiannacambel](https://kgptalkie.com/) on 29 August 2020  

---

## Feature Selection and CNN  

In this project, we are going to build a neural network to predict if a particular bank customer is satisfied or not. To do this, we are going to use **Convolutional Neural Networks (CNN)**. The dataset contains **370 features**. We are going to use **feature selection** to select the most relevant features and reduce the complexity of our model.  

---

## Dataset  

The dataset is an anonymized dataset containing a large number of numeric variables. The **TARGET** column is the variable to predict. It equals **1** for unsatisfied customers and **0** for satisfied customers.  

You can download the dataset from the following links:  
- [Kaggle: Santander Customer Satisfaction](https://www.kaggle.com/c/santander-customer-satisfaction/data)  
- [GitHub: Data Files for Feature Selection](https://github.com/laxmimerit/Data-Files-for-Feature-Selection)  

---

## Code Implementation  

### Install Required Libraries  

```bash
!pip install tensorflow
!pip install tensorflow-gpu
```

### Import Libraries  

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPool1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

print(tf.__version__)
# Output: 2.1.0
```

### Load and Preprocess Data  

```python
!git clone https://github.com/laxmimerit/Data-Files-for-Feature-Selection.git

data = pd.read_csv('train.csv')
data.head()
```

**Output:**  
| ID | var3 | var15 | ... | TARGET |
|----|------|-------|-----|--------|
| 1  | 2    | 23    | ... | 0      |
| 3  | 2    | 34    | ... | 0      |
| 4  | 2    | 23    | ... | 0      |
| 8  | 2    | 37    | ... | 0      |
| 10 | 2    | 39    | ... | 0      |

```python
data.shape  # (76020, 371)
```

### Feature Selection  

```python
X = data.drop(labels=['ID', 'TARGET'], axis=1)
y = data['TARGET']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
X_train.shape, X_test.shape  # ((60816, 369), (15204, 369))
```

#### Remove Constant and Quasi-Constant Features  

```python
filter = VarianceThreshold(0.01)
X_train = filter.fit_transform(X_train)
X_test = filter.transform(X_test)
X_train.shape, X_test.shape  # ((60816, 273), (15204, 273))
```

#### Remove Duplicate Features  

```python
X_train_T = X_train.T
X_test_T = X_test.T

X_train_T = pd.DataFrame(X_train_T)
X_test_T = pd.DataFrame(X_test_T)

duplicated_features = X_train_T.duplicated()
features_to_keep = [not index for index in duplicated_features]

X_train = X_train_T[features_to_keep].T
X_test = X_test_T[features_to_keep].T
X_train.shape, X_test.shape  # ((60816, 256), (15204, 256))
```

### Scale Data  

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Reshape for CNN  

```python
X_train = X_train.reshape(60816, 256, 1)
X_test = X_test.reshape(15204, 256, 1)
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
```

---

## Building the CNN  

```python
model = Sequential()
model.add(Conv1D(32, 3, activation='relu', input_shape=(256, 1)))
model.add(BatchNormalization())
model.add(MaxPool1D(2))
model.add(Dropout(0.3))

model.add(Conv1D(64, 3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool1D(2))
model.add(Dropout(0.5))

model.add(Conv1D(128, 3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool1D(2))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.summary()
```

**Model Summary:**  
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv1d (Conv1D)              (None, 254, 32)           128       
_________________________________________________________________
batch_normalization (BatchNo (None, 254, 32)           128       
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, 127, 32)           0         
_________________________________________________________________
dropout (Dropout)            (None, 127, 32)           0         
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 125, 64)           6208      
_________________________________________________________________
batch_normalization_1 (Batch (None, 125, 64)           256       
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 62, 64)            0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 62, 64)            0         
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 60, 128)           24704     
_________________________________________________________________
batch_normalization_2 (Batch (None, 60, 128)           512       
_________________________________________________________________
max_pooling1d_2 (MaxPooling1 (None, 30, 128)           0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 30, 128)           0         
_________________________________________________________________
flatten (Flatten)            (None, 3840)              0         
_________________________________________________________________
dense (Dense)                (None, 256)               983296    
_________________________________________________________________
dropout_3 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 257       
=================================================================
Total params: 1,015,489
Trainable params: 1,015,041
Non-trainable params: 448
```

---

## Training the Model  

```python
model.compile(optimizer=Adam(lr=0.00005), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=1)
```

**Training Output (Example):**  
```
Epoch 10/10
60816/60816 [==============================] - 111s 2ms/sample - loss: 0.1542 - accuracy: 0.9604 - val_loss: 0.1602 - val_accuracy: 0.9599
```

---

## Model Evaluation  

```python
history.history
```

**Output (Partial):**  
```python
{
  'accuracy': [0.95417327, 0.9592706, ...],
  'loss': [0.21693714527215763, 0.17656464240582592, ...],
  'val_accuracy': [0.9600763, 0.9600763, ...],
  'val_loss': [0.17092196812710614, 0.1765108920851371, ...]
}
```

---

## Plotting Learning Curve  

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

plot_learningCurve(history, 10)
```

---

## Conclusion  

We achieved an accuracy of **96%**. This demonstrates that **Convolutional Neural Networks (CNN)** with appropriate **feature selection** can build a powerful model for this dataset. Feature selection improves training speed, reduces model complexity, and enhances interpretability.  

**Source:** https://kgptalkie.com/bank-customer-satisfaction-prediction-using-cnn-and-feature-selection