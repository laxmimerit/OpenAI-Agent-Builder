https://kgptalkie.com/text-generation-using-tensorflow-keras-and-lstm

# Text Generation using Tensorflow, Keras and LSTM

Published by [georgiannacambel](https://kgptalkie.com) on 31 August 2020

## Automatic Text Generation

Automatic text generation is the generation of natural language texts by computer. It has applications in automatic documentation systems, automatic letter writing, automatic report generation, etc. In this project, we are going to generate words given a set of input words. We are going to train the LSTM model using William Shakespeare’s writings. The dataset is available [here](https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt).

## LSTM

Long Short-Term Memory (LSTM) networks are a modified version of recurrent neural networks, which makes it easier to remember past data in memory.

Generally LSTM is composed of a cell (the memory part of the LSTM unit) and three “regulators”, usually called gates, of the flow of information inside the LSTM unit: an input gate, an output gate and a forget gate.

Intuitively, the cell is responsible for keeping track of the dependencies between the elements in the input sequence.

The input gate controls the extent to which a new value flows into the cell, the forget gate controls the extent to which a value remains in the cell and the output gate controls the extent to which the value in the cell is used to compute the output activation of the LSTM unit.

The activation function of the LSTM gates is often the logistic sigmoid function.

There are connections into and out of the LSTM gates, a few of which are recurrent. The weights of these connections, which need to be learned during training, determine how the gates operate.

### Code: Importing Libraries

```python
%tensorflow_version 2.x
import tensorflow as tf
import string
import requests
```

### Code: Fetching Data

```python
response = requests.get('https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt')
response.text[:1500]
```

### Code: Cleaning Text

```python
def clean_text(doc):
  tokens = doc.split()
  table = str.maketrans('', '', string.punctuation)
  tokens = [w.translate(table) for w in tokens]
  tokens = [word for word in tokens if word.isalpha()]
  tokens = [word.lower() for word in tokens]
  return tokens
```

### Code: Preprocessing

```python
tokens = clean_text(data)
print(tokens[:50])
```

## Build LSTM Model and Prepare X and y

### Code: Importing Libraries

```python
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
```

### Code: Tokenizing and Preparing Data

```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)
sequences = np.array(sequences)
X, y = sequences[:, :-1], sequences[:,-1]
vocab_size = len(tokenizer.word_index) + 1
y = to_categorical(y, num_classes=vocab_size)
```

### Code: Model Definition

```python
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
model.summary()
```

### Code: Model Training

```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, batch_size=256, epochs=100)
```

### Code: Text Generation

```python
def generate_text_seq(model, tokenizer, text_seq_length, seed_text, n_words):
  text = []
  for _ in range(n_words):
    encoded = tokenizer.texts_to_sequences([seed_text])[0]
    encoded = pad_sequences([encoded], maxlen=text_seq_length, truncating='pre')
    y_predict = model.predict_classes(encoded)
    predicted_word = ''
    for word, index in tokenizer.word_index.items():
      if index == y_predict:
        predicted_word = word
        break
    seed_text = seed_text + ' ' + predicted_word
    text.append(predicted_word)
  return ' '.join(text)
```

### Code: Generating Text

```python
seed_text = lines[12343]
generate_text_seq(model, tokenizer, seq_length, seed_text, 100)
```

**Source:** https://kgptalkie.com/text-generation-using-tensorflow-keras-and-lstm