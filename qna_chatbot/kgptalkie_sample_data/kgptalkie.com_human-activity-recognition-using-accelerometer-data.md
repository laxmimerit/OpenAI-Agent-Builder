# Human Activity Recognition Using Accelerometer Data  
**Source:** [https://kgptalkie.com/human-activity-recognition-using-accelerometer-data](https://kgptalkie.com/human-activity-recognition-using-accelerometer-data)  

Published by [georgiannacambel](https://kgptalkie.com/human-activity-recognition-using-accelerometer-data) on 28 August 2020  

---

## Prediction of Human Activity  

In this project, we are going to use **accelerometer** data to train the model so that it can predict the human activity. We are going to use **2D Convolutional Neural Networks** to build the model.  

**Source:** [https://kgptalkie.com/human-activity-recognition-using-accelerometer-data](https://kgptalkie.com/human-activity-recognition-using-accelerometer-data)  

---

## Dataset  

**Dataset Link:**  
- [http://www.cis.fordham.edu/wisdm/dataset.php](http://www.cis.fordham.edu/wisdm/dataset.php)  
- [https://github.com/laxmimerit/Human-Activity-Recognition-Using-Accelerometer-Data-and-CNN](https://github.com/laxmimerit/Human-Activity-Recognition-Using-Accelerometer-Data-and-CNN)  

This WISDM dataset contains data collected through controlled, laboratory conditions. The total number of examples is **1,098,207**. The dataset contains six different labels: **Downstairs, Jogging, Sitting, Standing, Upstairs, Walking**.  

---

## Code Setup  

We will be using **tensorflow-keras** to build the CNN.  

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
print(tf.__version__)
# Output: 2.1.0
```

Other libraries used:  
- **pandas** for reading the dataset  
- **numpy** for array operations  
- **matplotlib.pyplot** for visualization  
- **train_test_split** from **sklearn** for splitting data  
- **StandardScaler** and **LabelEncoder** from **sklearn** for preprocessing  

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
```

---

## Load and Process the Dataset  

### Data Preprocessing  

If we try to read this data directly using `pd.read_csv()`, we will get an error. So we will read the file and preprocess it:  

```python
file = open('WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt')
lines = file.readlines()

processedList = []
for i, line in enumerate(lines):
    try:
        line = line.split(',')
        last = line[5].split(';')[0]
        last = last.strip()
        if last == '':
            break
        temp = [line[0], line[1], line[2], line[3], line[4], last]
        processedList.append(temp)
    except:
        print('Error at line number: ', i)
```

**Output:**  
```
Error at line number:  281873
Error at line number:  281874
Error at line number:  281875
```

### Create DataFrame  

```python
columns = ['user', 'activity', 'time', 'x', 'y', 'z']
data = pd.DataFrame(data=processedList, columns=columns)
data.head()
```

**Sample Output:**  
| user | activity | time        | x         | y         | z         |
|------|----------|-------------|-----------|-----------|-----------|
| 33   | Jogging  | 49105962326000 | -0.6946377 | 12.680544 | 0.50395286 |
| 33   | Jogging  | 49106062271000 | 5.012288  | 11.264028 | 0.95342433 |

---

## Data Balancing  

### Activity Distribution  

```python
data['activity'].value_counts()
```

**Output:**  
```
Walking       137375
Jogging       129392
Upstairs       35137
Downstairs     33358
Sitting         4599
Standing        3555
```

### Balance the Data  

We take the first 3555 lines for each activity:  

```python
Walking = df[df['activity']=='Walking'].head(3555).copy()
Jogging = df[df['activity']=='Jogging'].head(3555).copy()
Upstairs = df[df['activity']=='Upstairs'].head(3555).copy()
Downstairs = df[df['activity']=='Downstairs'].head(3555).copy()
Sitting = df[df['activity']=='Sitting'].head(3555).copy()
Standing = df[df['activity']=='Standing'].copy()

balanced_data = pd.DataFrame()
balanced_data = balanced_data.append([Walking, Jogging, Upstairs, Downstairs, Sitting, Standing])
balanced_data.shape
# Output: (21330, 4)
```

---

## Label Encoding  

```python
label = LabelEncoder()
balanced_data['label'] = label.fit_transform(balanced_data['activity'])
```

**Class Mapping:**  
```python
label.classes_
# Output: ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']
```

---

## Standardization  

```python
X = balanced_data[['x', 'y', 'z']]
y = balanced_data['label']

scaler = StandardScaler()
X = scaler.fit_transform(X)

scaled_X = pd.DataFrame(data=X, columns=['x', 'y', 'z'])
scaled_X['label'] = y.values
```

---

## Frame Preparation  

```python
import scipy.stats as stats

Fs = 20
frame_size = Fs * 4  # 80
hop_size = Fs * 2    # 40

def get_frames(df, frame_size, hop_size):
    N_FEATURES = 3
    frames = []
    labels = []
    for i in range(0, len(df) - frame_size, hop_size):
        x = df['x'].values[i: i + frame_size]
        y = df['y'].values[i: i + frame_size]
        z = df['z'].values[i: i + frame_size]
        label = stats.mode(df['label'][i: i + frame_size])[0][0]
        frames.append([x, y, z])
        labels.append(label)
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURE段
    labels = np.asarray(labels)
    return frames, labels

X, y = get_frames(scaled_X, frame_size, hop_size)
X.shape, y.shape
# Output: ((532, 80, 3), (532,))
```

---

## Train-Test Split  

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
X_train.shape, X_test.shape
# Output: ((425, 80, 3), (107, 80, 3))
```

---

## 2D CNN Model  

```python
model = Sequential()
model.add(Conv2D(16, (2, 2), activation='relu', input_shape=X_train[0].shape))
model.add(Dropout(0.1))
model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=1)
```

**Training Output:**  
```
Epoch 10/10
425/425 [==============================] - 0s 303us/sample - loss: 0.2385 - accuracy: 0.9388 - val_loss: 0.2269 - val_accuracy: 0.8972
```

---

## Model Evaluation  

### Plot Learning Curve  

```python
def plot_learningCurve(history, epochs):
    epoch_range = range(1, epochs+1)
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

## Confusion Matrix  

```python
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

y_pred = model.predict_classes(X_test)
mat = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(conf_mat=mat, class_names=label.classes_, show_normed=True, figsize=(7,7))
```

**Source:** [https://kgptalkie.com/human-activity-recognition-using-accelerometer-data](https://kgptalkie.com/human-activity-recognition-using-accelerometer-data)  

---

## Save Model  

```python
model.save_weights('model.h5')
```