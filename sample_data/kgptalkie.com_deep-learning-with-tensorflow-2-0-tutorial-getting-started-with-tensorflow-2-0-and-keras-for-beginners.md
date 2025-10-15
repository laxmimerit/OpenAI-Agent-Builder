https://kgptalkie.com/deep-learning-with-tensorflow-2-0-tutorial-getting-started-with-tensorflow-2-0-and-keras-for-beginners

# Deep Learning with Tensorflow 2.0 Tutorial – Getting Started with Tensorflow 2.0 and Keras for Beginners

**Source:** https://kgptalkie.com/deep-learning-with-tensorflow-2-0-tutorial-getting-started-with-tensorflow-2-0-and-keras-for-beginners

Published by  
georgiannacambel  
on  
27 August 2020

## Classification using Fashion MNIST dataset

### What is TensorFlow?

TensorFlow is one of the best libraries to implement deep learning. It is a software library for numerical computation of mathematical expressions, using data flow graphs. Nodes in the graph represent mathematical operations, while the edges represent the multidimensional data arrays (tensors) that flow between them. It was created by [Google](https://www.google.com) and tailored for [Machine Learning](https://en.wikipedia.org/wiki/Machine_learning). In fact, it is being widely used to develop solutions with [Deep Learning](https://en.wikipedia.org/wiki/Deep_learning).

#### TensorFlow architecture works in three parts:
1. Preprocessing the data
2. Build the model
3. Train and estimate the model

---

## Why Every Data Scientist should Learn Tensorflow 2.x and not Tensorflow 1.x?

- **API Cleanup**
- **Eager execution**
- **No more globals**
- **Functions, not sessions** (`session.run()`)
- **Use Keras layers and models to manage variables**
- **It is faster**
- **It takes less space**
- **More consistent**

For more information, visit these links:
- [Google I/O](https://www.youtube.com/watch?v=lEljKc9ZtU8)
- [TensorFlow Releases](https://github.com/tensorflow/tensorflow/releases)
- [Effective TensorFlow 2 Guide](https://www.tensorflow.org/guide/effective_tf2)

---

## Installation

You can install TensorFlow by running the following command in the [Anaconda admin shell](https://docs.anaconda.com/anaconda/install/):

```bash
pip install tensorflow
```

If you have a [GPU](https://en.wikipedia.org/wiki/Graphics_processing_unit) in your machine, use this command to install the GPU version:

```bash
pip install tensorflow-gpu
```

---

## Fashion MNIST dataset

We are going to use the Fashion Mnist dataset. It consists of:
- 60,000 training examples
- 10,000 test examples

All images are 28×28 grayscale and belong to 10 categories:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

You can download the dataset from [here](https://github.com/zalandoresearch/fashion-mnist).

### Code Example

```python
import tensorflow as tf
from tensorflow import keras

print(tf.__version__)  # 2.1.0
```

The necessary Python libraries:
- `numpy` for basic operations
- `matplotlib.pyplot` for visualization
- `pandas` for reading datasets

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

### Loading the Dataset

```python
mnist = keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

### Data Exploration

- `X_train.shape`: `(60000, 28, 28)`
- `X_test.shape`: `(10000, 28, 28)`
- `np.max(X_train)`: `255`
- `np.mean(X_train)`: `72.94`

Class names:
```python
class_names = ['top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
```

---

## Data Preprocessing

Normalize pixel values to `[0, 1]`:

```python
X_train = X_train / 255.0
X_test = X_test / 255.0
```

---

## Build the Model with TF 2.0

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()
```

### Model Summary

```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten_1 (Flatten)          (None, 784)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 128)               100480    
_________________________________________________________________
dense_3 (Dense)              (None, 10)                1290      
=================================================================
Total params: 101,770
Trainable params: 101,770
Non-trainable params: 0
_________________________________________________________________
```

---

## Model Compilation and Training

```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)
```

### Training Output

```
Epoch 10/10
60000/60000 [==============================] - 5s 79us/sample - loss: 0.2374 - accuracy: 0.9120
```

---

## Model Evaluation

```python
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")  # 0.8841
```

---

## Predictions

```python
from sklearn.metrics import accuracy_score

y_pred = model.predict_classes(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")  # 0.8841
```

### Example Prediction

```python
pred = model.predict(X_test)
np.argmax(pred[0])  # 9 (ankle boot)
np.argmax(pred[1])  # 2 (pullover)
```

---

## Overfitting

- **Training Accuracy**: 91.2%
- **Test Accuracy**: 88.41%

To avoid overfitting, consider using **Convolutional Neural Networks (CNNs)**.

**Source:** https://kgptalkie.com/deep-learning-with-tensorflow-2-0-tutorial-getting-started-with-tensorflow-2-0-and-keras-for-beginners