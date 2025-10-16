# Word Embedding and NLP with TF2.0 and Keras on Twitter Sentiment Data

**Source:** https://kgptalkie.com/word-embedding-and-nlp-with-tf2-0-and-keras-on-twitter-sentiment-data

Published by [georgiannacambel](https://kgptalkie.com/word-embedding-and-nlp-with-tf2-0-and-keras-on-twitter-sentiment-data) on 13 September 2020

## Word Embedding and Sentiment Analysis

### What is Word Embedding?

Natural Language Processing (NLP) refers to computer systems designed to understand human language. Human language, like English or Hindi, consists of words and sentences, and NLP attempts to extract information from these sentences.

Machine learning and deep learning algorithms only take numeric input. So how do we convert text to numbers?

A word embedding is a learned representation for text where words that have the same meaning have a similar representation. Embeddings translate large sparse vectors into a lower-dimensional space that preserves semantic relationships.

Word embeddings is a technique where individual words of a domain or language are represented as real-valued vectors in a lower-dimensional space. The sparse matrix problem with BOW is solved by mapping high-dimensional data into a lower-dimensional space. The lack of meaningful relationship issue of BOW is solved by placing vectors of semantically similar items close to each other. This way, words that have similar meaning have similar distances in the vector space as shown below. “king is to queen as man is to woman” encoded in the vector space as well as verb tense and country and their capitals are encoded in low-dimensional space preserving the semantic relationships.

## Dataset

The dataset used is the [sentiment140 dataset](https://www.kaggle.com/kazanova/sentiment140/data#). It contains 1,600,000 tweets extracted using the Twitter API. We are going to use 4000 tweets for training our model. The tweets have been annotated (0 = negative, 1 = positive) and can be used to detect sentiment.

You can download the modified dataset from [here](https://kgptalkie.com/word-embedding-and-nlp-with-tf2-0-and-keras-on-twitter-sentiment-data).

**Watch Full Video:** [here](https://kgptalkie.com/word-embedding-and-nlp-with-tf2-0-and-keras-on-twitter-sentiment-data)

## Code Implementation

### Importing Libraries

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Activation, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
```

### Loading Data

```python
df = pd.read_csv('twitter4000.csv')
df.head()
```

| twitts                                      | sentiment |
|-------------------------------------------|-----------|
| is bored and wants to watch a movie any sugge… | 0         |
| back in miami. waiting to unboard ship      | 0         |
| @misskpey awwww dnt dis brng bak memoriessss, … | 0         |
| ughhh i am so tired blahhhhhhhhh            | 0         |
| @mandagoforth me bad! It’s funny though. Zacha… | 0         |

### Sentiment Distribution

```python
df['sentiment'].value_counts()
```

```
1    2000
0    2000
Name: sentiment, dtype: int64
```

### Text and Labels

```python
text = df['twitts'].tolist()
y = df['sentiment']
```

### Tokenization

```python
token = Tokenizer()
token.fit_on_texts(text)
vocab_size = len(token.word_index) + 1
```

### Encoding Text

```python
encoded_text = token.texts_to_sequences(text)
print(encoded_text[:30])
```

### Padding Sequences

```python
max_length = 120
X = pad_sequences(encoded_text, maxlen=max_length, padding='post')
print(X.shape)  # (4000, 120)
```

### Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)
```

### Model Definition

```python
vec_size = 300

model = Sequential()
model.add(Embedding(vocab_size, vec_size, input_length=max_length))
model.add(Conv1D(64, 8, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))
```

### Model Compilation and Training

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
```

**Output:**

```
Epoch 1/5
3200/3200 [==============================] - 8s 2ms/sample - loss: 0.6937 - acc: 0.4919 - val_loss: 0.6870 - val_acc: 0.5188
Epoch 2/5
3200/3200 [==============================] - 6s 2ms/sample - loss: 0.6588 - acc: 0.6212 - val_loss: 0.6328 - val_acc: 0.6425
Epoch 3/5
3200/3200 [==============================] - 5s 2ms/sample - loss: 0.5100 - acc: 0.7625 - val_loss: 0.6255 - val_acc: 0.6787
Epoch 4/5
3200/3200 [==============================] - 6s 2ms/sample - loss: 0.3110 - acc: 0.8763 - val_loss: 0.7330 - val_acc: 0.6925
Epoch 5/5
3200/3200 [==============================] - 6s 2ms/sample - loss: 0.1663 - acc: 0.9394 - val_loss: 0.7949 - val_acc: 0.6775
```

### Model Prediction

```python
def get_encoded(x):
    x = token.texts_to_sequences(x)
    x = pad_sequences(x, maxlen=max_length, padding='post')
    return x

x = ['worst services. will not come again']
model.predict_classes(get_encoded(x))  # array([[0]])

x = ['thank you for watching']
model.predict_classes(get_encoded(x))  # array([[1]])
```

## Conclusion

We can increase the accuracy of the model by training it on the entire dataset of 1,600,000 tweets. We can even use more pre-processing techniques like checking for spelling mistakes, repeated letters, etc.

**Source:** https://kgptalkie.com/word-embedding-and-nlp-with-tf2-0-and-keras-on-twitter-sentiment-data