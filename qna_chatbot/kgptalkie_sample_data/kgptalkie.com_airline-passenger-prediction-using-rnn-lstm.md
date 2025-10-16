https://kgptalkie.com/airline-passenger-prediction-using-rnn-lstm

# Airline Passenger Prediction using RNN – LSTM  
**Source:** https://kgptalkie.com/airline-passenger-prediction-using-rnn-lstm  

Published by georgiannacambel on 29 August 2020  

## Prediction of Number of Passengers for an Airline Using LSTM  
In this project, we are going to build a model to predict the number of passengers in an airline. To do so, we are going to use **Recurrent Neural Networks**, more precisely **Long Short Term Memory**.

---

## Recurrent Neural Network  
Neural Networks are a set of algorithms that closely resemble the human brain and are designed to recognize patterns.  

**Recurrent Neural Network** is a generalization of feedforward neural networks that has an internal memory.  
- RNN is recurrent in nature as it performs the same function for every input of data.  
- The output of the current input depends on the past computation.  
- After producing the output, it is copied and sent back into the recurrent network.  
- For making a decision, it considers the current input and the output learned from the previous input.  
- In other neural networks, all inputs are independent. In RNN, all inputs are related to each other.

---

## Long Short Term Memory  
**Long Short-Term Memory (LSTM)** networks are a modified version of recurrent neural networks, which makes it easier to remember past data in memory.  

### Key Components of LSTM  
- **Cell**: The memory part of the LSTM unit.  
- **Gates**: Three regulators of the flow of information inside the LSTM unit:  
  - **Input gate**: Controls the extent to which a new value flows into the cell.  
  - **Forget gate**: Controls the extent to which a value remains in the cell.  
  - **Output gate**: Controls the extent to which the value in the cell is used to compute the output activation.  

- The activation function of the LSTM gates is often the logistic sigmoid function.  
- Connections into and out of the LSTM gates are recurrent. The weights of these connections determine how the gates operate.

---

## Dataset  
This dataset provides monthly totals of a US airline passengers from 1949 to 1960. The dataset has 2 columns:  
- `month`: Contains the month of the year.  
- `passengers`: Contains the total number of passengers travelled on that particular month.  

You can download the dataset from [here](https://kgptalkie.com/airline-passenger-prediction-using-rnn-lstm).  

---

## Implementation Steps  

### Install Required Libraries  
```bash
!pip install tensorflow
!pip install tensorflow-gpu
```

### Import Libraries  
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
```

### Load and Preprocess Data  
```python
dataset = pd.read_csv('AirPassengers.csv')
dataset = dataset['#Passengers']
dataset = np.array(dataset).reshape(-1, 1)
```

### Scale the Data  
```python
scaler = MinMaxScaler()
dataset = scaler.fit_transform(dataset)
print(dataset.min(), dataset.max())  # Output: (0.0, 1.0)
```

### Split Data into Training and Testing Sets  
```python
train_size = 100
test_size = 44
train = dataset[0:train_size, :]
test = dataset[train_size:144, :]
```

### Create Training and Testing Datasets  
```python
def get_data(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 1
X_train, y_train = get_data(train, look_back)
X_test, y_test = get_data(test, look_back)
```

### Reshape Data for LSTM  
```python
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
print(X_train.shape)  # Output: (98, 1, 1)
```

### Build the LSTM Model  
```python
model = Sequential()
model.add(LSTM(5, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()
```

### Train the Model  
```python
model.fit(X_train, y_train, epochs=50, batch_size=1)
```

### Evaluate the Model  
```python
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = np.array(y_test).reshape(-1, 1)
y_test = scaler.inverse_transform(y_test)
```

### Visualize Results  
```python
plt.figure(figsize=(14, 5))
plt.plot(y_test, label='Real number of passengers')
plt.plot(y_pred, label='Predicted number of passengers')
plt.ylabel('# passengers')
plt.legend()
plt.show()
```

---

**Source:** https://kgptalkie.com/airline-passenger-prediction-using-rnn-lstm  

As we can see, the actual results and the predicted results follow the same trend. Our model is predicting the number of passengers with good accuracy.  

**Source:** https://kgptalkie.com/airline-passenger-prediction-using-rnn-lstm