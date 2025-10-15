# https://kgptalkie.com/classify-dog-or-cat-by-the-help-of-convolutional-neural-networkcnn

## Classify Dog or Cat by the help of Convolutional Neural Network(CNN)

Published by *pasqualebrownlow* on **4 September 2020**

### Use of Dropout and Batch Normalization in 2D CNN on Dog Cat Image Classification in TensorFlow 2.0

We are going to predict cat or dog by the help of **Convolutional neural network**. I have taken the dataset from Kaggle:  
https://www.kaggle.com/tongpython/cat-and-dog

In this dataset, there are two classes: cats and dogs. We have to predict whether the cat or dog by the help of CNN algorithm.

### Deep Learning

Deep Learning is a subfield of **machine learning** concerned with algorithms inspired by the structure and function of the brain called an **artificial neural network**.

### Convolutional Neural Network (ConvNet/CNN)

A **Convolutional Neural Network (ConvNet/CNN)** is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. The pre-processing required in a ConvNet is much lower as compared to other classification algorithms.

### What is Dropout

Dropout is a technique where randomly selected neurons are ignored during training. They are “dropped-out” randomly. This means that their contribution to the activation of downstream neurons is temporally removed on the forward pass and any weight updates are not applied to the neuron on the backward pass.

### What is Batch Normalization

It is a technique which is designed to automatically standardize the inputs to a layer in a deep learning neural network.  
**Example**: We have four features having different units after applying batch normalization it comes in similar units.

By normalizing the output of neurons, the activation function will only receive inputs close to zero. Batch normalization ensures a non-vanishing gradient. Normalization brings all the inputs centered around 0. This way, there is not much change in each layer input. So, layers in the network can learn from the back-propagation simultaneously, without waiting for the previous layer to learn. This fastens up the training of networks.

### VGG16 Model

**VGG16** is a convolution neural net (CNN) architecture that was used to win ILSVR (Imagenet) competition in 2014. It is considered to be one of the excellent vision model architectures to date. The most unique thing about VGG16 is that instead of having a large number of hyper-parameters, they focused on having convolution layers of 3×3 filters with a stride of 1 and always used the same padding and Maxpool layer of 2×2 filters with a stride of 2. It follows this arrangement of convolution and max pool layers consistently throughout the whole architecture. In the end, it has 2 FC (fully connected layers) followed by a softmax for output.

---

## Download Data from GitHub and Model Building

We are going to use **TensorFlow 2.3** (the latest version) to build the model. You can install TensorFlow by running this command:

```bash
!pip install tensorflow-gpu==2.3.0-rc0
```

### Importing Necessary Libraries

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, ZeroPadding2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD

print(tf.__version__)
```

Output:
```
2.3.0
```

### Data Preparation

```python
import numpy as np
import matplotlib.pyplot as plt
!git clone https://github.com/laxmimerit/dog-cat-full-dataset.git
```

Output:
```
Cloning into 'dog-cat-full-dataset'...
remote: Enumerating objects: 25027, done.
remote: Total 25027 (delta 0), reused 0 (delta 0), pack-reused 25027
Receiving objects: 100% (25027/25027), 541.62 MiB | 37.27 MiB/s, done.
Resolving deltas: 100% (5/5), done.
Checking out files: 100% (25001/25001), done.
```

```python
test_data_dir = '/content/dog-cat-full-dataset/data/test'
train_data_dir = '/content/dog-cat-full-dataset/data/train'
img_width = 32
img_height = 32
batch_size = 20
```

### Image Data Augmentation

```python
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(
    directory=train_data_dir,
    target_size=(img_width, img_height),
    classes=['dogs', 'cats'],
    class_mode='binary',
    batch_size=batch_size
)
```

Output:
```
Found 20000 images belonging to 2 classes.
```

```python
validation_generator = datagen.flow_from_directory(
    directory=test_data_dir,
    target_size=(32, 32),
    classes=['dogs', 'cats'],
    class_mode='binary',
    batch_size=batch_size
)
```

Output:
```
Found 5000 images belonging to 2 classes.
```

```python
data_generator = train_generator * batch_size
len(train_generator) * batch_size
```

Output:
```
20000
```

---

## Build CNN Model

### Sequential Model

A `Sequential()` function is the easiest way to build a model in Keras. It allows you to build a model layer by layer.

### 2D Convolution Layer

A 2D convolution layer means that the input of the convolution operation is three-dimensional, for example, a color image that has a value for each pixel across three layers: red, blue, and green. However, it is called a 2D convolution because the movement of the filter across the image happens in two dimensions.

### Rectified Linear Unit (ReLU)

The **Rectified Linear Unit (ReLU)** is the most commonly used activation function in deep learning models. The function returns 0 if it receives any negative input, but for any positive value $ x $, it returns that value back. So it can be written as:
$$ f(x) = \max(0, x) $$

### Padding

To stop the problem of shrinkage of data, we use the concept called **Padding**. It has two types: `valid` and `same`.

### Max Pooling

**Max pooling** is a sample-based discretization process. The objective is to down-sample an input representation, reducing its dimensionality and allowing for assumptions to be made about features contained in the sub-regions binned. This helps with **over-fitting** by providing an abstracted form of the representation.

### Flattening

**Flattening** is converting the data into a 1-dimensional array for inputting it to the next layer. We flatten the output of the convolutional layers to create a single long feature vector.

### Sigmoid Function

The **Sigmoid function** takes a value as input and outputs another value between 0 and 1. It is non-linear and easy to work with when constructing a neural network model. The good part about this function is that it is continuously differentiable over different values of $ z $ and has a fixed output range.

---

### CNN Model Implementation

```python
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer='he_uniform', input_shape=(img_width, img_height, 3)))
model.add(MaxPool2D(2,2))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='sigmoid'))
```

### Compile the Model

```python
opt = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
```

### Train the Model

```python
history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=len(train_generator),
    epochs=5,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    verbose=1
)
```

**Source:** https://kgptalkie.com/classify-dog-or-cat-by-the-help-of-convolutional-neural-networkcnn

---

## Implement First 3 Blocks of VGG16 Model

```python
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer='he_uniform', input_shape=(img_width, img_height, 3)))
model.add(MaxPool2D(2,2))

model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer='he_uniform'))
model.add(MaxPool2D(2,2))

model = Sequential()
model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer='he_uniform'))
model.add(MaxPool2D(2,2))

model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='sigmoid'))
```

**Source:** https://kgptalkie.com/classify-dog-or-cat-by-the-help-of-convolutional-neural-networkcnn

---

## Batch Normalization and Dropout

```python
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer='he_uniform', input_shape=(img_width, img_height, 3)))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.2))

model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))

model = Sequential()
model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
```

**Source:** https://kgptalkie.com/classify-dog-or-cat-by-the-help-of-convolutional-neural-networkcnn

---

## Training with Batch Normalization and Dropout

```python
opt = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    verbose=1
)
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

In this blog, you definitely learn that **Dropout** and **Batch Normalisation** have a great impact on Deep Neural Networks.  

**Source:** https://kgptalkie.com/classify-dog-or-cat-by-the-help-of-convolutional-neural-networkcnn