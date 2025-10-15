https://kgptalkie.com/breast-cancer-detection-using-cnn

# Breast Cancer Detection Using CNN

Published by [pasqualebrownlow](https://kgptalkie.com/breast-cancer-detection-using-cnn) on 5 September 2020

## Breast Cancer Detection Using CNN in Python

**Breast cancer** is the most commonly occurring cancer in women and the second most common cancer overall. There were over 2 million new cases in 2018, making it a significant health problem in present days.

The key challenge in **breast cancer detection** is to classify tumors as **malignant** or **benign**. Malignant refers to cancer cells that can invade and kill nearby tissue and spread to other parts of your body. Unlike cancerous tumor (malignant), Benign does not spread to other parts of the body and is safe somehow. Deep neural network techniques can be used to improve the accuracy of early diagnosis significantly.

### Deep Learning

Deep Learning is a subfield of **machine learning** concerned with algorithms inspired by the structure and function of the brain called an **artificial neural network**.

### Convolutional Neural Network (ConvNet/CNN)

A **Convolutional Neural Network (ConvNet/CNN)** is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. The pre-processing required in a ConvNet is much lower as compared to other classification algorithms.

## What is Dropout

**Dropout** is a technique where randomly selected neurons are ignored during training. They are “dropped-out” randomly. This means that their contribution to the activation of downstream neurons is temporally removed on the forward pass and any weight updates are not applied to the neuron on the backward pass.

## What is Batch Normalization

It is a technique which is designed to automatically standardize the inputs to a layer in a deep learning neural network.

**Example**: We have four features having different unit after applying batch normalization it comes in similar unit. By Normalizing the output of neurons the activation function will only receive inputs close to zero. Batch normalization ensures a non vanishing gradient.

## Implementation

We are going to use **tensorflow** 2.3 to build the model. You can install tensorflow by running this command:

```python
!pip install tensorflow-gpu==2.3.0-rc0
```

### Importing Libraries

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPool1D
from tensorflow.keras.optimizers import Adam

print(tf.__version__)
# 2.3.0
```

### Data Loading and Preprocessing

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
cancer = datasets.load_breast_cancer()

# View dataset description
print(cancer.DESCR)
```

### Dataset Description

**Source:** https://kgptalkie.com/breast-cancer-detection-using-cnn

**Data Set Characteristics:**

- **Number of Instances:** 569
- **Number of Attributes:** 30 numeric, predictive attributes and the class

**Attribute Information:**

- radius (mean of distances from center to points on the perimeter)
- texture (standard deviation of gray-scale values)
- perimeter
- area
- smoothness (local variation in radius lengths)
- compactness (perimeter^2 / area - 1.0)
- concavity (severity of concave portions of the contour)
- concave points (number of concave portions of the contour)
- symmetry 
- fractal dimension ("coastline approximation" - 1)

The mean, standard error, and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features.

**Class Distribution:**
- 212 - Malignant
- 357 - Benign

### Data Preparation

```python
# Convert to DataFrame
X = pd.DataFrame(data = cancer.data, columns=cancer.feature_names)
y = cancer.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape for CNN
X_train = X_train.reshape(455, 30, 1)
X_test = X_test.reshape(114, 30, 1)
```

## Model Architecture

```python
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(30,1)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.summary()
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
_________________________________________________________________
```

## Model Training

```python
model.compile(optimizer=Adam(lr=0.00005), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=1)
```

### Training Output

```
Epoch 46/50
15/15 [==============================] - 0s 6ms/step - loss: 0.1054 - accuracy: 0.9560 - val_loss: 0.1064 - val_accuracy: 0.9649
Epoch 47/50
15/15 [==============================] - 0s 6ms/步 - loss: 0.1373 - accuracy: 0.9473 - val_loss: 0.1074 - val_accuracy: 0.9649
...
Epoch 50/50
15/15 [==============================] - 0s 6ms/step - loss: 0.0927 - accuracy: 0.9648 - val_loss: 0.1047 - val_accuracy: 0.9649
```

## Learning Curve

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

plot_learningCurve(history, 50)
```

### Model Evaluation

- **Validation accuracy** is always greater than train accuracy, which means our model is not overfitting.
- **Validation loss** is also very lower than training loss, so unless and until validation loss goes above than the training loss, we can keep training our model.

We have successfully created our program to detect breast cancer using **Deep neural network**. We are able to classify cancer effectively with our CNN technique.