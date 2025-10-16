[https://kgptalkie.com/deep-learning-with-tensorflow-2-0-tutorial-building-your-first-ann-with-tensorflow-2-0](https://kgptalkie.com/deep-learning-with-tensorflow-2-0-tutorial-building-your-first-ann-with-tensorflow-2-0)

# Deep Learning with TensorFlow 2.0 Tutorial – Building Your First ANN with TensorFlow 2.0

**Source:** [https://kgptalkie.com/deep-learning-with-tensorflow-2-0-tutorial-building-your-first-ann-with-tensorflow-2-0](https://kgptalkie.com/deep-learning-with-tensorflow-2-0-tutorial-building-your-first-ann-with-tensorflow-2-0)

Published by ignacioberrios on 27 August 2020

## Objective

Our objective for this code is to build an Artificial Neural Network (ANN) for a classification problem using **TensorFlow** and **Keras** libraries. We will learn how to build a neural network model using **TensorFlow** and **Keras**, then analyze our model using different accuracy metrics.

---

## What is ANN?

**Artificial Neural Networks (ANN)** is a supervised learning system built of a large number of simple elements, called **neurons or perceptrons**. Each neuron can make simple decisions, and feeds those decisions to other neurons, organized in interconnected layers.

---

## What is Activation Function?

In artificial neural networks, the **activation function** of a node defines the output of that node given an input or set of inputs. A standard integrated circuit can be seen as a digital network of activation functions that can be “ON” (1) or “OFF” (0), depending on input. This is similar to the behavior of the linear perceptron in neural networks.

If we do not apply an activation function, the output signal would simply be a simple linear function. A linear function is just a polynomial of one degree.

---

## Types of Activation Function

- **Sigmoid**
- **Tanh**
- **ReLU**
- **LeakyReLU**
- **SoftMax**

---

## What is Back Propagation?

In **backpropagation**, we update the parameters of the model with respect to the **loss function**. The loss function can be **cross entropy** for **classification** problems and **root mean squared error** for **regression** problems.

Our objective is to **minimize loss** of our model. To minimize loss, we calculate the **gradient of loss** with respect to the **parameters** of the model and try to minimize this gradient. While minimizing the gradient, we update the weights of our model. This process is known as **back propagation**.

---

## Steps for Building Your First ANN

1. **Data Preprocessing**
2. **Add input layer**
3. **Random weight initialization**
4. **Add hidden layers**
5. **Select optimizer, loss, and performance metrics**
6. **Compile the model**
7. **Use model.fit to train the model**
8. **Evaluate the model**
9. **Adjust optimization parameters or model if needed**

---

## Data Preprocessing

It is better to preprocess data before giving it to any neural net model. Data should be **normally distributed** (Gaussian distribution), so that the model performs well.

If our data is not normally distributed, that means there is **skewness** in data. To remove skewness of data, we can take the logarithm of data. By using the log function, we can remove skewness of data.

After removing skewness of data, it is better to scale the data so that all values are at the same scale. We can either use **MinMaxScaler** or **StandardScaler**.

**StandardScaler** is better to use since by using it, the mean and variance of our data is now 0 and 1 respectively. That is, now our data is in the form of **N(0,1)** (Gaussian distribution with mean 0 and variance 1).

---

## Layers

### Adding **input layer**

According to the size of our input, we add the number of input layers.

### Adding **hidden layers**

We can add as many hidden layers as needed. If we want our model to be complex, a large number of hidden layers can be added, and for a simple model, the number of hidden layers can be small.

### Adding **output layer**

In a **classification problem**, the size of the output layer depends on the number of classes.

In a **regression problem**, the size of the output layer is one.

---

## Weight Initialization

- The **mean** of the weights should be zero.
- The **variance** of the weights should stay the same across every layer.

---

## Optimizers

### Gradient Descent

Gradient descent is a first-order optimization algorithm dependent on the first-order derivative of a loss function. It calculates which way the weights should be altered so that the function can reach a minima.

### Stochastic Gradient Descent (SGD)

It’s a variant of Gradient Descent. It tries to update the model’s parameters more frequently. In this, the model parameters are altered after computation of loss on each training example.

### Mini-Batch Gradient Descent

It’s best among all the variations of gradient descent algorithms. It is an improvement on both SGD and standard gradient descent. It updates the model parameters after every batch.

### Adagrad

It is gradient descent with an adaptive learning rate. In this, the learning rate decays for parameters in proportion to their update history (more updates means more decay).

---

## Losses

- **Cross entropy** for **Classification** problems.
- **Root mean squared error** for **Regression** problems.

---

## Accuracy Metrics

$$
\text{Accuracy} = \frac{TP + TN}{TP + FP + TN + FN}
$$

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

---

## Installing Libraries

```bash
# pip install tensorflow==2.0.0-rc0
# pip install tensorflow-gpu==2.0.0-rc0
```

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
print(tf.__version__)
```

---

## Importing Necessary Libraries

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
```

---

## Data Preparation

```python
dataset = pd.read_csv('Customer_Churn_Modelling.csv')
dataset.head()
```

```python
X = dataset.drop(labels=['CustomerId', 'Surname', 'RowNumber', 'Exited'], axis=1)
y = dataset['Exited']
```

---

## Label Encoding

```python
from sklearn.preprocessing import LabelEncoder
label1 = LabelEncoder()
X['Geography'] = label1.fit_transform(X['Geography'])
label = LabelEncoder()
X['Gender'] = label.fit_transform(X['Gender'])
```

---

## One-Hot Encoding

```python
X = pd.get_dummies(X, drop_first=True, columns=['Geography'])
```

---

## Scaling Data

```python
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

## Build ANN

```python
model = Sequential()
model.add(Dense(X.shape[1], activation='relu', input_dim=X.shape[1]))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

---

## Compile the Model

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

---

## Train the Model

```python
model.fit(X_train, y_train.to_numpy(), batch_size=10, epochs=10, verbose=1)
```

---

## Evaluate the Model

```python
model.evaluate(X_test, y_test.to_numpy())
```

---

## Predictions

```python
y_pred = model.predict_classes(X_test)
```

---

## Confusion Matrix and Accuracy

```python
from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
```

---

## Summary

In this notebook, we have implemented a classifier using an artificial neural network. We built the model using **TensorFlow** and **Keras**. We checked the accuracy using **Accuracy metrics** and **Confusion matrix**. Accuracy for the model was **85.2%** on test data.

**Source:** [https://kgptalkie.com/deep-learning-with-tensorflow-2-0-tutorial-building-your-first-ann-with-tensorflow-2-0](https://kgptalkie.com/deep-learning-with-tensorflow-2-0-tutorial-building-your-first-ann-with-tensorflow-2-0)