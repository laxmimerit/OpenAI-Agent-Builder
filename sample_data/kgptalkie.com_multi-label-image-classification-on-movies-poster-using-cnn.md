https://kgptalkie.com/multi-label-image-classification-on-movies-poster-using-cnn

# Multi-Label Image Classification on Movies Poster using CNN

Published by [georgiannacambel](https://kgptalkie.com/multi-label-image-classification-on-movies-poster-using-cnn) on 7 September 2020

## Multi-Label Image Classification in Python

In this project, we are going to train our model on a set of labeled movie posters. The model will predict the genres of the movie based on the movie poster. We will consider a set of **25 genres**. Each poster can have more than one genre.

### What is multi-label classification?

In multi-label classification, the training set is composed of instances each associated with a set of labels, and the task is to predict the label sets of unseen instances through analyzing training instances with known label sets.

Multi-label classification and the strongly related problem of multi-output classification are variants of the classification problem where multiple labels may be assigned to each instance.

In the multi-label problem, there is no constraint on how many of the classes the instance can be assigned to.

Multiclass classification makes the assumption that each sample is assigned to one and only one label whereas Multilabel classification assigns to each sample a set of target labels.

## Dataset

**Dataset Link:** [https://www.cs.ccu.edu.tw/~wtchu/projects/MoviePoster/index.html](https://www.cs.ccu.edu.tw/~wtchu/projects/MoviePoster/index.html)

This dataset was collected from the IMDB website. One poster image was collected from one (mostly) Hollywood movie released from 1980 to 2015. Each poster image is associated with a movie as well as some metadata like ID, genres, and box office. The ID of each image is set as its file name.

You can even clone the github repository using the following command to get the dataset:

```bash
!git clone https://github.com/laxmimerit/Movies-Poster_Dataset.git
```

## Tensorflow Installation

We are going to use [tensorflow](https://www.tensorflow.org/) to build the model. You can install tensorflow by running this command. If your machine has a GPU you can use the second command:

```bash
!pip install tensorflow
!pip install tensorflow-gpu
```

Watch Full Video Here: [https://kgptalkie.com/multi-label-image-classification-on-movies-poster-using-cnn](https://kgptalkie.com/multi-label-image-classification-on-movies-poster-using-cnn)

The necessary Python libraries are imported here:

- **Tensorflow** is used to build the neural network.
- We have even imported all the layers required to build the model from **keras**.
- **numpy** is used to perform basic array operations
- **pandas** for loading and manipulating the data.
- **pyplot** from matplotlib is used to visualize the results.
- **train_test_split** is used to split the data into training and testing datasets.
- **tqdm** is a progress bar library with good support for nested loops and Jupyter/IPython notebooks.

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
print(tf.__version__)
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
```

Here we are reading the dataset into a pandas dataframe using `pd.read_csv()`. The dataset contains 7254 rows and 27 columns.

```python
data = pd.read_csv('/content/Movies-Poster_Dataset/train.csv')
data.shape
```

Output:
```
(7254, 27)
```

```python
data.head()
```

| Id       | Genre              | Action | Adventure | Animation | Biography | Comedy | Crime | Documentary | Drama | Family | Fantasy | History | Horror | Music | Musical | Mystery | N/A | News | Reality-TV | Romance | Sci-Fi | Short | Sport | Thriller | War | Western |
|----------|--------------------|--------|-----------|-----------|-----------|--------|-------|-------------|-------|--------|---------|---------|--------|-------|---------|---------|-----|------|------------|---------|--------|-------|-------|----------|---|---------|
| tt0086425| ['Comedy', 'Drama']| 0      | 0         | 0         | 0         | 1      | 0     | 0           | 1     | 0      | 0       | 0       | 0      | 0     | 0       | 0       | 0   | 0    | 0          | 0       | 0      | 0     | 0     | 0         | 0 | 1       |
| tt0085549| ['Drama', 'Romance', 'Music'] | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

The images in the dataset are of different sizes. To process the images we need to convert them into a fixed size.

```python
img_width = 350
img_height = 350

X = []
for i in tqdm(range(data.shape[0])):
  path = '/content/Movies-Poster_Dataset/Images/' + data['Id'][i] + '.jpg'
  img = image.load_img(path, target_size=(img_width, img_height, 3))
  img = image.img_to_array(img)
  img = img/255.0
  X.append(img)

X = np.array(X)
```

Output:
```
100%|██████████| 7254/7254 [00:33<00:00, 216.79it/s]
```

```python
X.shape
```

Output:
```
(7254, 350, 350, 3)
```

Now we will see an image from `X`.

```python
plt.imshow(X[1])
```

We can get the `Genre` of the above image from `data`.

```python
data['Genre'][1]
```

Output:
```
"['Drama', 'Romance', 'Music']"
```

Now we will prepare the dataset. We have already got the feature space in `X`. Now we will get the target in `y`. For that, we will drop the `Id` and `Genre` columns from data.

```python
y = data.drop(['Id', 'Genre'], axis = 1)
y = y.to_numpy()
y.shape
```

Output:
```
(7254, 25)
```

Now we will split the data into training and testing set with the help of `train_test_split()`.

```python
test_size = 0.15
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.15)
```

```python
X_train[0].shape
```

Output:
```
(350, 350, 3)
```

## Build CNN

A `Sequential()` model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.

### Model Architecture

```python
model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape = X_train[0].shape))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))

model.add(Conv2D(32, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.4))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(25, activation='sigmoid'))
model.summary()
```

### Model Summary

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 348, 348, 16)      448       
_________________________________________________________________
batch_normalization (BatchNo (None, 348, 348, 16)      64        
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 174, 174, 16)      0         
_________________________________________________________________
dropout (Dropout)            (None, 174, 174, 16)      0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 172, 172, 32)      4640      
_________________________________________________________________
batch_normalization_1 (Batch (None, 172, 172, 32)      128       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 86, 86, 32)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 86, 86, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 84, 84, 64)        18496     
_________________________________________________________________
batch_normalization_2 (Batch (None, 84, 84, 64)        256       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 42, 42, 64)        0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 42, 42, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 40, 40, 128)       73856     
_________________________________________________________________
batch_normalization_3 (Batch (None, 40, 40, 128)       512       
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 20, 20, 128)       0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 20, 20, 128)       0         
_________________________________________________________________
flatten (Flatten)            (None, 51200)             0         
_________________________________________________________________
dense (Dense)                (None, 128)               6553728   
_________________________________________________________________
batch_normalization_4 (Batch (None, 128)               512       
_________________________________________________________________
dropout_4 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               16512     
_________________________________________________________________
batch_normalization_5 (Batch (None, 128)               512       
_________________________________________________________________
dropout_5 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 25)                3225      
=================================================================
Total params: 6,672,889
Trainable params: 6,671,897
Non-trainable params: 992
_________________________________________________________________
```

Now we will compile and fit the model. We will use 5 epochs to train the model. An epoch is an iteration over the entire data provided.

```python
model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
```

Output:
```
Train on 6165 samples, validate on 1089 samples
Epoch 1/5
6165/6165 [==============================] - 720s 117ms/sample - loss: 0.6999 - accuracy: 0.6431 - val_loss: 0.5697 - val_accuracy: 0.7450
Epoch 2/5
6165/6165 [==============================] - 721s 117ms/sample - loss: 0.3138 - accuracy: 0.8869 - val_loss: 0.2531 - val_accuracy: 0.9071
Epoch 3/5
6165/6165 [==============================] - 718s 116ms/sample - loss: 0.2615 - accuracy: 0.9057 - val_loss: 0.2407 - val_accuracy: 0.9078
Epoch 4/5
6165/6165 [==============================] - 720s 117ms/sample - loss: 0.2525 - accuracy: 0.9085 - val_loss: 0.2388 - val_accuracy: 0.9096
Epoch 5/5
6165/6165 [==============================] - 723s 117ms/sample - loss: 0.2468 - accuracy: 0.9100 - val_loss: 0.2362 - val_accuracy: 0.9119
```

Now we will visualize the results.

```python
def plot_learningCurve(history, epoch):
    # Plot training & validation accuracy values
    epoch_range = range(1, epoch+1)
    plt.plot(epoch_range, history.history['accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
    # Plot training & validation loss values
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

plot_learningCurve(history, 5)
```

We can see that the validation accuracy is more than the training accuracy and the validation loss is less than the training loss. Hence the model is not overfitting.

## Testing of model

Now we are going to test our model by giving it a new image. We will pre-process the image in the same way as we have pre-processed the images in the training and testing dataset. We will normalize them and convert them into a size of 350×350.

```python
classes = data.columns[2:]
print(classes)
y_prob = model.predict(img)
top3 = np.argsort(y_prob[0])[:-4:-1]
for i in range(3):
  print(classes[top3[i]])
```

Output:
```
Index(['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
       'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 'Horror',
       'Music', 'Musical', 'Mystery', 'N/A', 'News', 'Reality-TV', 'Romance',
       'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western'],
      dtype='object')
Drama
Action
Adventure
```

As you can see for the above movie poster our model has selected 3 genres which are **Drama**, **Action**, and **Adventure**.

**Source:** [https://kgptalkie.com/multi-label-image-classification-on-movies-poster-using-cnn](https://kgptalkie.com/multi-label-image-classification-on-movies-poster-using-cnn)