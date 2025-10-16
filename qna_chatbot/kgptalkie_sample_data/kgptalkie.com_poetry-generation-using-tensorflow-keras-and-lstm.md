https://kgptalkie.com/poetry-generation-using-tensorflow-keras-and-lstm

# Poetry Generation Using Tensorflow, Keras, and LSTM  
**Published by**  
**9 August 2020**  

## What is RNN  
Recurrent Neural Networks (RNNs) are State of the Art algorithms that can memorize/remember previous inputs in memory when processing sequential data.  

These loops make RNNs seem mysterious, but they are essentially multiple copies of the same network passing messages to successors.  

## Different Types of RNNs  
RNNs handle various tasks:  
- **Image Classification**  
- **Sequence Output** (e.g., image captioning)  
- **Sequence Input** (e.g., sentiment analysis)  
- **Sequence Input and Output** (e.g., machine translation)  
- **Synced Sequence Input and Output** (e.g., video classification)  

## The Problem of RNNs: Long-Term Dependencies  
### Vanishing Gradient  
If the partial derivative of error is less than 1, multiplying it with the learning rate results in minimal changes between iterations.  

### Exploding Gradient  
Occurs when gradients are assigned excessively high importance. This can be mitigated by truncating or squashing gradients.  

## Long Short-Term Memory (LSTM) Networks  
LSTMs are a special type of RNN designed to avoid long-term dependency issues. They naturally remember information over long periods.  

## Sequence Generation Scheme  
### Let’s Code  
```python
import tensorflow as tf
import string
import requests
import pandas as pd

response = requests.get('https://raw.githubusercontent.com/laxmimerit/poetry-data/master/adele.txt')
data = response.text.splitlines()
```

### Prepare Training Data  
```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences

token = Tokenizer()
token.fit_on_texts(data)
encoded_text = token.texts_to_sequences(data)

vocab_size = len(token.word_counts) + 1
```

### Padding and Splitting Sequences  
```python
max_length = 20
sequences = pad_sequences(datalist, maxlen=max_length, padding='pre')
X = sequences[:, :-1]
y = sequences[:, -1]
y = to_categorical(y, num_classes=vocab_size)
```

### LSTM Model Training  
```python
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, batch_size=32, epochs=50)
```

### Model Summary  
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 19, 50)            69800     
_________________________________________________________________
lstm (LSTM)                  (None, 19, 100)           60400     
_________________________________________________________________
lstm_1 (LSTM)                (None, 100)               80400     
_________________________________________________________________
dense (Dense)                (None, 100)               10100     
_________________________________________________________________
dense_1 (Dense)              (None, 1396)              140996    
=================================================================
Total params: 361,696
Trainable params: 361,696
Non-trainable params: 0
_________________________________________________________________
```

## Poetry Generation  
```python
def generate_poetry(seed_text, n_lines):
    for i in range(n_lines):
        text = []
        for _ in range(poetry_length):
            encoded = token.texts_to_sequences([seed_text])
            encoded = pad_sequences(encoded, maxlen=seq_length, padding='pre')
            y_pred = np.argmax(model.predict(encoded), axis=-1)
            predicted_word = ""
            for word, index in token.word_index.items():
                if index == y_pred:
                    predicted_word = word
                    break
            seed_text = seed_text + ' ' + predicted_word
            text.append(predicted_word)
        seed_text = text[-1]
        text = ' '.join(text)
        print(text)

seed_text = 'i love you'
generate_poetry(seed_text, 5)
```

**Source:** https://kgptalkie.com/poetry-generation-using-tensorflow-keras-and-lstm  

Watch Full Course Here:  
[http://bitly.com/nlp_intro](http://bitly.com/nlp_intro)