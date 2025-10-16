https://kgptalkie.com/google-stock-price-prediction-using-rnn-lstm

# Google Stock Price Prediction using RNN – LSTM

**Published by**  
georgiannacambel  
**on**  
24 August 2020  
24 August 2020

## Prediction of Google Stock Price using RNN

In this article, we are going to predict the opening price of the stock given the highest, lowest, and closing price for that particular day by using RNN-LSTM.

**Ref:**  
[https://colah.github.io/posts/2015-08-Understanding-LSTMs/](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## What is RNN?

Recurrent Neural Networks are the first of its kind State of the Art algorithms that can  
memorize/remember previous inputs in memory  
when a huge set of Sequential data is given to it.

Recurrent Neural Network is a generalization of feedforward neural network that has an internal memory.  
RNN is recurrent in nature as it performs the same function for every input of data while the output of the current input depends on the past one computation.  
After producing the output, it is copied and sent back into the recurrent network. For making a decision, it considers the current input and the output that it has learned from the previous input.  
In other neural networks, all the inputs are independent of each other. But in RNN, all the inputs are related to each other.  
These loops make recurrent neural networks seem kind of mysterious. However, if you think a bit more, it turns out that they aren’t all that different than a normal neural network.  
A recurrent neural network can be thought of as multiple copies of the same network, each passing a message to a successor.

## Different types of RNNs

Some examples are:

- **One to one**  
  It is used for **Image Classification**. Here input is a single image and output is a single label of the category the image belongs.

- **One to many** (Sequence output)  
  It is used for **Image Captioning**. Here the input is an image and output is a group of words which is the caption for the image.

- **Many to One** (Sequence input)  
  It is used for **Sentiment Analysis**. Here a given sentence which is a group of words is classified as expressing positive or negative sentiment which is a single output.

- **Many to Many** (Sequence input and sequence output)  
  It is **Machine Translation**. A RNN reads a sentence in English and then outputs a sentence in French.

- **Synced Many to Many** (Synced sequence input and output)  
  It is used for **Video Classification** where we wish to label each frame of the video.

## The Problem of Long-Term Dependencies

### Vanishing Gradient

Information travels through the neural network from input neurons to the output neurons, while the error is calculated and propagated back through the network to update the weights.  
During the training, the cost function (e) compares your outcomes to your desired output.  
If the partial derivation of error is less than 1, then when it gets multiplied with the learning rate which is also very less won’t generate a big change when compared with previous iteration.  
For the vanishing gradient problem, the further you go through the network, the lower your gradient is and the harder it is to train the weights, which has a domino effect on all of the further weights throughout the network.

### Exploding Gradient

We speak of Exploding Gradients when the algorithm assigns a stupidly high importance to the weights, without any reason.  
Exploding gradients are a problem where large error gradients accumulate and result in very large updates to neural network model weights during training.  
This has the effect of your model being unstable and unable to learn from your training data.  
But fortunately, this problem can be easily solved if you truncate or squash the gradients.

## Long Short Term Memory (LSTM) Networks

Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of **learning long-term dependencies**.  
Generally LSTM is composed of a cell (the memory part of the LSTM unit) and three “regulators”, usually called gates, of the flow of information inside the LSTM unit: an **input gate**, an **output gate** and a **forget gate**.  
Intuitively, the cell is responsible for keeping track of the dependencies between the elements in the input sequence.  
The input gate controls the extent to which a new value flows into the cell, the forget gate controls the extent to which a value remains in the cell and the output gate controls the extent to which the value in the cell is used to compute the output activation of the LSTM unit.  
The activation function of the LSTM gates is often the logistic sigmoid function.  
There are connections into and out of the LSTM gates, a few of which are recurrent. The weights of these connections, which need to be learned during training, determine how the gates operate.  
LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn!

## Dataset

You can download the dataset from [here](https://kgptalkie.com/google-stock-price-prediction-using-rnn-lstm).  
The data used in this notebook is from 19th August, 2004 to 7th October, 2019. The dataset consists of 7 columns which contain the **date, opening price, highest price, lowest price, closing price, adjusted closing price and volume** of share for each day.

## Steps to build stock prediction model

1. **Data Preprocessing**  
2. **Building the RNN**  
3. **Making the prediction and visualization**

We will read the data for first 60 days and then predict for the 61st day. Then we will hop ahead by one day and read the next chunk of data for next sixty days.

### Necessary Python Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
```

### Data Loading and Inspection

```python
data = pd.read_csv('GOOG.csv', date_parser=True)
data.tail()
```

**Output:**

```
Date        Open        High         Low       Close  Adj Close     Volume
3804 2019-09-30 1220.97 1226.00 1212.30 1219.00 1219.00 1404100
3805 2019-10-01 1219.00 1231.23 1203.58 1205.10 1205.10 1273500
3806 2019-10-02 1196.98 1196.98 1171.29 1176.63 1176.63 1615100
3807 2019-10-03 1180.00 1189.06 1162.43 1187.83 1187.83 1621200
3808 2019-10-04 1191.89 1211.44 1189.17 1209.00 1209.00 1021092
```

### Splitting Data into Training and Testing Sets

```python
data_training = data[data['Date'] < '2019-01-01'].copy()
data_test = data[data['Date'] >= '2019-01-01'].copy()
```

### Data Preprocessing

```python
data_training = data_training.drop(['Date', 'Adj Close'], axis=1)
scaler = MinMaxScaler()
data_training = scaler.fit_transform(data_training)
```

### Preparing Training Data

```python
X_train = []
y_train = []
for i in range(60, data_training.shape[0]):
    X_train.append(data_training[i-60:i])
    y_train.append(data_training[i, 0])
    
X_train, y_train = np.array(X_train), np.array(y_train)
```

**Shape of X_train:**  
```python
X_train.shape  # (3557, 60, 5)
```

### Building LSTM Model

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

regressor = Sequential()
regressor.add(LSTM(units=60, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 5)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=60, activation='relu', return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=80, activation='relu', return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=120, activation='relu'))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))
regressor.summary()
```

**Model Summary:**

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lstm (LSTM)                  (None, 60, 60)            15840     
_________________________________________________________________
dropout (Dropout)            (None, 60, 60)            0         
_________________________________________________________________
lstm_1 (LSTM)                (None, 60, 60)            29040     
_________________________________________________________________
dropout_1 (Dropout)          (None, 60, 60)            0         
_________________________________________________________________
lstm_2 (LSTM)                (None, 60, 80)            45120     
_________________________________________________________________
dropout_2 (Dropout)          (None, 60, 80)            0         
_________________________________________________________________
lstm_3 (LSTM)                (None, 120)               96480     
_________________________________________________________________
dropout_3 (Dropout)          (None, 120)               0         
_________________________________________________________________
dense (Dense)                (None, 1)                 121       
=================================================================
Total params: 186,601
Trainable params: 186,601
Non-trainable params: 0
_________________________________________________________________
```

### Compiling and Training the Model

```python
regressor.compile(optimizer='adam', loss='mean_squared_error')
regressor.fit(X_train, y_train, epochs=50, batch_size=32)
```

**Training Output:**

```
Epoch 45/50
3557/3557 [==============================] - 26s 7ms/sample - loss: 6.8088e-04
Epoch 46/50
3557/3557 [==============================] - 25s 7ms/sample - loss: 6.0968e-04
Epoch 47/50
3557/3557 [==============================] - 25s 7ms/sample - loss: 6.6604e-04
Epoch 48/50
3557/3557 [==============================] - 25s 7ms/sample - loss: 6.2150e-04
Epoch 49/50
3557/3557 [==============================] - 25s 7ms/sample - loss: 6.4292e-04
Epoch 50/50
3557/3557 [==============================] - 25s 7ms/sample - loss: 6.3066e-04
```

### Preparing Test Data

```python
past_60_days = data_training.tail(60)
df = past_60_days.append(data_test, ignore_index=True)
df = df.drop(['Date', 'Adj Close'], axis=1)
inputs = scaler.transform(df)
```

### Preparing Test Data for Prediction

```python
X_test = []
y_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i])
    y_test.append(inputs[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)
```

**Shape of X_test and y_test:**

```python
X_test.shape, y_test.shape  # ((192, 60, 5), (192,))
```

### Making Predictions

```python
y_pred = regressor.predict(X_test)
scale = 1 / 8.18605127e-04
y_pred = y_pred * scale
y_test = y_test * scale
```

### Visualization

```python
plt.figure(figsize=(14, 5))
plt.plot(y_test, color='red', label='Real Google Stock Price')
plt.plot(y_pred, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
```

**Source:** [https://kgptalkie.com/google-stock-price-prediction-using-rnn-lstm](https://kgptalkie.com/google-stock-price-prediction-using-rnn-lstm)