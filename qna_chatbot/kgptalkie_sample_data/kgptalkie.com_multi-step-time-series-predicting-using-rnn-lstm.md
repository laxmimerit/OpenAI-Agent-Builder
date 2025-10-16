# Multi-step-Time-series-predicting using RNN LSTM  
**Source:** [https://kgptalkie.com/multi-step-time-series-predicting-using-rnn-lstm](https://kgptalkie.com/multi-step-time-series-predicting-using-rnn-lstm)  

## Published by  
**berryedelson**  
**28 August 2020**  

---

## Household Power Consumption Prediction using RNN-LSTM  

### Problem Statement  
Given power consumption data for the previous week, predict the power consumption for the next week.  

### Dataset  
- **Download dataset:** [https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip](https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip)  
- **Details:** [https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)  

### Dataset Description  
- **Timeframe:** December 2006 to November 2010  
- **Variables:**  
  - `global_active_power`: Total active power consumed (kilowatts)  
  - `global_reactive_power`: Total reactive power consumed (kilowatts)  
  - `voltage`: Average voltage (volts)  
  - `global_intensity`: Average current intensity (amps)  
  - `sub_metering_1`: Active energy for kitchen (watt-hours)  
  - `sub_metering_2`: Active energy for laundry (watt-hours)  
  - `sub_metering_3`: Active energy for climate control systems (watt-hours)  

---

## Why Use Deep Learning Algorithms?  
- **Advantages:**  
  - Handles large volumes of data  
  - Captures complex patterns and seasonality  
  - Achieves **maximum accuracy** through neural network tuning  

### LSTM Overview  
- **LSTM (Long Short-Term Memory)**: Overcomes RNN limitations by using **forget gates**, **input gates**, and **output gates**.  

### LSTM Cell Breakdown  
1. **Forget Gate**  
   $ f = \text{Sigmoid}(W \cdot [h_{t-1}, X_t] + b) $  
2. **Input Gate**  
   $ I = \text{Sigmoid}(W \cdot [h_{t-1}, X_t] + b) $  
   $ II = \tanh(W \cdot [h_{t-1}, X_t] + b) $  
3. **Update Step**  
   $ C_t = C_{t-1} \cdot f + I \cdot II $  
4. **Output Gate**  
   $ i = \text{Sigmoid}(W \cdot [h_{t-1}, X_t] + b) $  
   $ h_t = i \cdot \tanh(C_t) $  

---

## Implementation Steps  

### 1. Import Libraries  
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
```

### 2. Load and Preprocess Data  
```python
data = pd.read_csv('household_power_consumption.txt', sep=';', parse_dates=True, low_memory=False)
data['date_time'] = data['Date'].str.cat(data['Time'], sep=' ')
data.drop(['Date', 'Time'], axis=1, inplace=True)
data.set_index('date_time', inplace=True)
```

### 3. Handle Missing Values  
```python
data.replace('?', np.nan, inplace=True)
data = data.astype('float')
```

### 4. Resample Data (Daily Aggregation)  
```python
data = dataset.resample('D').sum()
```

---

## Exploratory Data Analysis  

### Plotting Features  
```python
fig, ax = plt.subplots(figsize=(18, 18))
for i in range(len(data.columns)):
    plt.subplot(len(data.columns), 1, i+1)
    name = data.columns[i]
    plt.plot(data[name])
    plt.title(name, y=0, loc='right')
    plt.yticks([])
plt.show()
```

### Year-wise Analysis  
```python
years = ['2007', '2008', '2009', '2010']
fig, ax = plt.subplots(figsize=(18, 18))
for i in range(len(years)):
    plt.subplot(len(years), 1, i+1)
    year = years[i]
    active_power_data = data[str(year)]
    plt.plot(active_power_data['Global_active_power'])
    plt.title(str(year), y=0, loc='left')
plt.show()
```

---

## Model Building  

### Train-Test Split  
```python
data_train = data.loc[:'2009-12-31', :]['Global_active_power']
data_test = data['2010']['Global_active_power']
```

### Data Normalization  
```python
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X_train = x_scaler.fit_transform(X_train)
y_train = y_scaler.fit_transform(y_train)
```

### LSTM Model  
```python
reg = Sequential()
reg.add(LSTM(units=200, activation='relu', input_shape=(7, 1)))
reg.add(Dense(7))
reg.compile(loss='mse', optimizer='adam')
reg.fit(X_train, y_train, epochs=100)
```

---

## Model Evaluation  

### Predict and Inverse Transform  
```python
y_pred = reg.predict(X_test)
y_pred = y_scaler.inverse_transform(y_pred)
y_true = y_scaler.inverse_transform(y_test)
```

### Evaluation Metrics  
```python
def evaluate_model(y_true, y_predicted):
    scores = []
    for i in range(y_true.shape[1]):
        mse = mean_squared_error(y_true[:, i], y_predicted[:, i])
        rmse = np.sqrt(mse)
        scores.append(rmse)
    total_score = 0
    for row in range(y_true.shape[0]):
        for col in range(y_predicted.shape[1]):
            total_score += (y_true[row, col] - y_predicted[row, col])**2
    total_score = np.sqrt(total_score / (y_true.shape[0] * y_predicted.shape[1]))
    return total_score, scores

evaluate_model(y_true, y_pred)
```

---

## Conclusion  
- **RMSE:** ~598 watts  
- **Standard Deviation:** ~710 watts  
- **Model Performance:** RMSE < Standard Deviation → Model performs **well**.  

**Source:** [https://kgptalkie.com/multi-step-time-series-predicting-using-rnn-lstm](https://kgptalkie.com/multi-step-time-series-predicting-using-rnn-lstm)