https://kgptalkie.com/words-embedding-using-glove-vectors

# Words Embedding using GloVe Vectors

**Published by**  
berryedelson  
**on**  
28 August 2020

---

## NLP Tutorial – GloVe Vectors Embedding with TF2.0 and Keras

### What is GloVe?

GloVe stands for **global vectors for word representation**. It is an **unsupervised learning algorithm** developed by Stanford for generating word embeddings by aggregating a **global word-word co-occurrence matrix** from a corpus. The resulting embeddings show interesting **linear substructures** of the word in vector space.

**Ref:**  
- [Glove Vectors](https://nlp.stanford.edu/projects/glove/)  
- [Common Crawl (840B tokens)](http://nlp.stanford.edu/data/glove.840B.300d.zip)  
- [Twitter (2B tweets)](http://nlp.stanford.edu/data/glove.twitter.27B.zip)

---

### Watch Full Video:
**GloVe** is one of the text encodings patterns. If you have the NLP project in your hand then **GloVe** or **Word2Vec** are important topics.

But the first question is: **What is GloVe?**  
**GloVe**: Global Vectors Word Representation.

We know that a **machine can understand only the numbers**. The machine will not understand what is mean by “I am Indian”. So to transform this into numbers there are some mechanisms. We can use the **One-Hot-Encoding** also. In **One-Hot-Encoding** we create a matrix of the **n-dimension** of the words against a vocabulary. **Vocabulary** is a list of words and then we put 1 in the row where word of our text matches and the rest of the places are 0.

But there is a problem: if you see these are just the word representation in 0 and 1 but there is no linking or direction in the numbers or words. Or we can not find the distance between the 2 words or 2 sentences using this encoding method.

But the second question is: **what we are going to do by using this distance between the 2 words**? Yes, definitely question is important. So just take a simple example: we have one problem statement and we have 10 documents and we have to find the given text is the best matching to which of the 10 documents. Here we can not just use some word matching algorithms and say that these specific words are matching in a few of the document. **It Is because there may be some other document that has similar words and not the exact word as per query text**.

So to find the similarity between the words we need a vector that will give us the word representation in different dimensions and then we can compare this word vector with another word vector and find the distance between them.

To accomplish this task we need to find the vector representation of the word. And one of the best ways to find word representation in vector-matrix is **GLOVE**.

---

## Notebook Setup

### Importing Libraries

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Activation, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
import numpy as np
from numpy import array
import pandas as pd
from sklearn.model_selection import train_test_split
```

### Reading Dataset

```python
df = pd.read_csv('twitter4000.csv')
df.head()
```

| twitts                                      | sentiment |
|---------------------------------------------|-----------|
| is bored and wants to watch a movie any sugge… | 0         |
| back in miami. waiting to unboard ship        | 0         |
| @misskpey awwww dnt dis brng bak memoriessss, … | 0         |
| ughhh i am so tired blahhhhhhhhh              | 0         |
| @mandagoforth me bad! It’s funny though. Zacha… | 0         |

---

## Preprocessing and Cleaning

Here, we are doing the text processing where we are performing below steps:

- Expanding the contracted words or tokens
- Removing Email
- Removing URLs and HTML tags
- Removing ‘RT’ retweet tags
- Replacing all non-alphabets values with null

### Dictionary for Contractions

```python
contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    # ... (other contractions)
    "n": "and"
}
```

### Cleaning Function

```python
import re

def get_clean_text(x):
    if type(x) is str:
        x = x.lower()
        for key in contractions:
            x = x.replace(key, contractions[key])
        x = re.sub(r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)', '', x)
        x = re.sub(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', x)
        x = re.sub('RT', "", x)
        x = re.sub('[^A-Z a-z]+', '', x)
        x = ' '.join([t for t in x.split() if t not in rare])
    return x
```

### Applying Cleaning

```python
df['twitts'] = df['twitts'].apply(lambda x: get_clean_text(x))
```

---

## GloVe Vectors

### Loading GloVe Vectors

```python
glove_vectors = dict()

file = open('glove.twitter.27B.200d.txt', encoding='utf-8')
for line in file:
    values = line.split()
    word = values[0]
    vectors = np.asarray(values[1:])
    glove_vectors[word] = vectors
file.close()
```

### Creating Word Vector Matrix

```python
word_vector_matrix = np.zeros((vocab_size, 200))
for word, index in token.word_index.items():
    vector = glove_vectors.get(word)
    if vector is not None:
        word_vector_matrix[index] = vector
    else:
        print(word)
```

---

## Model Building

### Splitting Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)
```

### Building the Model

```python
vec_size = 200

model = Sequential()
model.add(Embedding(vocab_size, vec_size, input_length=max_length, weights=[word_vector_matrix], trainable=False))
model.add(Conv1D(64, 8, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
```

### Training the Model

```python
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))
```

**Output:**

```
Epoch 1/30
3200/3200 [==============================] - 10s 3ms/sample - loss: 0.7431 - acc: 0.5000 - val_loss: 0.6940 - val_acc: 0.4975
...
Epoch 30/30
3200/3200 [==============================] - 3s 810us/sample - loss: 0.4668 - acc: 0.7866 - val_loss: 0.5312 - val_acc: 0.7400
```

---

## Prediction

### Encoding New Text

```python
def get_encode(x):
    x = get_clean_text(x)
    x = token.texts_to_sequences(x)
    x = pad_sequences(x, maxlen=max_length, padding='post')
    return x

get_encode(["i hi how are you isn't"])
```

**Output:**

```python
array([[  1, 318,  77,  37,   7,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0]])
```

### Predicting on Text

```python
model.predict_classes(get_encode(['thank you for watching']))
```

**Output:**

```python
array([[1]])
```

---

## Summary

- Firstly, we have loaded the dataset using **pandas**.
- After loading the dataset, we have cleaned the dataset using a function `get_clean_text()`.
- Then using `Tokenizer` we have tokenized the entire text corpus.
- We have used GloVe vectors to create a dictionary and then converted it to a weight matrix (used the same during model training).
- Here we have used loss function as **binary_crossentropy** and metric as **‘accuracy’**.

**Source:** https://kgptalkie.com/words-embedding-using-glove-vectors