https://kgptalkie.com/sentiment-classification-using-bert

# Sentiment Classification Using BERT

**Published by**: berryedelson  
**Date**: 23 August 2020

---

## An Introduction to BERT

### Problem Statement

We will use the **IMDB Movie Reviews Dataset**, where based on the given review, we have to classify the sentiments of that particular review as **positive** or **negative**.

---

## Introduction

Chatbots, virtual assistants, and dialog agents typically classify queries into specific intents to generate coherent responses. **Intent classification** is a classification problem that predicts the intent label for any given user query. It is usually a **multi-class classification** problem, where the query is assigned one unique label.

### Example Queries and Intents

- **Query**: "how much does the limousine service cost within Pune"  
  **Intent**: "groundfare"

- **Query**: "what kind of ground transportation is available in Pune"  
  **Intent**: "ground_service"

- **Query**: "I want to fly from Delhi at 8:38 am and arrive in Pune at 11:10 in the morning"  
  **Intent**: "flight"

- **Query**: "show me the costs and times for flights from Delhi to Pune"  
  **Intent**: "airfare+flight_time"

These examples illustrate the ambiguity in intent labeling. Any addition of misleading words can cause multiple intents to be present in the same query.

---

## What is Transformer?

### Why Transformers?

To understand **Transformers**, we first need to understand the **attention mechanism**.

The **attention mechanism** enables Transformers to have extremely long-term memory. A Transformer model can "attend" or "focus" on all previous tokens that have been generated.

**Recurrent Neural Networks (RNNs)** are also capable of looking at previous inputs. However, the attention mechanism does not suffer from short-term memory issues. RNNs can theoretically access information arbitrarily far in the past, but in practice, they struggle with retaining that information (related to the **vanishing gradient problem**). This is also true for **Gated Recurrent Units (GRUs)** and **Long Short-Term Memory (LSTMs)**, although they have a larger capacity for longer-term memory.

The attention mechanism, in theory, can have an infinite window to reference from, making it capable of using the entire context of the story while generating text.

### Reference

For more information, read the article:  
[Attention is All You Need!](https://arxiv.org/abs/1706.03762)

---

## What is BERT?

**Bidirectional Encoder Representations from Transformers (BERT)** is a technique for **NLP (Natural Language Processing)** pre-training developed by **Google**. BERT was created and published in 2018 by **Jacob Devlin** and his colleagues from Google. Google is leveraging BERT to better understand user searches.

### Key Features of BERT

- BERT is designed to pre-train **deep bidirectional representations** from unlabeled text by jointly conditioning on both **left and right context** in all layers.
- The pre-trained BERT model can be **fine-tuned** with just one additional output layer to create **state-of-the-art models** for tasks like **question answering** and **language inference**, without substantial task-specific architecture modifications.

### Key Points to Remember

- BERT is a trained **Transformer Encoder stack**:
  - **Base version**: 12 encoder layers
  - **Large version**: 24 encoder layers
- BERT encoders have larger **feedforward networks**:
  - **Base**: 768 nodes
  - **Large**: 1024 nodes
- BERT has more **attention heads**:
  - **Base**: 12
  - **Large**: 16
- BERT was trained on **Wikipedia** and **Book Corpus** (a dataset containing +10,000 books of different genres).

---

## Why BERT?

Proper language representation is key for general-purpose language understanding by machines. **Context-free models** like **word2vec** or **GloVe** generate a single word embedding for each word in the vocabulary. For example, the word **"bank"** would have the same representation in **"bank deposit"** and **"riverbank"**.

**Contextual models**, like BERT, generate a representation of each word based on the other words in the sentence. BERT captures these relationships in a **bidirectional** way.

BERT is built upon recent work and clever ideas in pre-training contextual representations, including:

- **Semi-supervised Sequence Learning**
- **Generative Pre-Training**
- **ELMo**
- **OpenAI Transformer**
- **ULMFit**
- **Transformer**

Although these models are unidirectional or shallowly bidirectional, **BERT is fully bidirectional**.

We will use BERT to extract high-quality language features from the **ATIS query text data** and fine-tune BERT on a specific task (classification) with its own data to produce **state-of-the-art predictions**.

---

## Additional Reading

- [Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Understanding searches better than ever before](https://www.blog.google/products/search/search-language-understanding-bert/)
- [Good Resource to Read More About the BERT](http://jalammar.github.io/illustrated-bert/)
- [Visual Guide to Using BERT](http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)

---

## What is ktrain?

**ktrain** is a library to help build, train, debug, and deploy neural networks in the **Keras** deep learning framework. It uses **tf.keras** in **TensorFlow** instead of standalone Keras. Inspired by the **fastai** library, **ktrain** allows you to:

- Estimate an optimal learning rate for your model using a **learning rate finder**
- Employ learning rate schedules such as **triangular learning rate policy**, **1cycle policy**, and **SGDR**
- Use pre-canned models for **text classification** (e.g., **NBSVM**, **fastText**, **GRU with pre-trained word embeddings**) and **image classification** (e.g., **ResNet**, **Wide Residual Networks**, **Inception**)
- Load and preprocess text and image data from various formats
- Inspect misclassified data points to improve your model
- Leverage a simple prediction API for saving and deploying models and preprocessing steps

**ktrain GitHub**: [https://github.com/amaiya/ktrain](https://github.com/amaiya/ktrain)

---

## Notebook Setup

```bash
!pip install ktrain
```

```python
import tensorflow as tf
import pandas as pd
import numpy as np
import ktrain
from ktrain import text
```

```python
tf.__version__
'2.1.0'
```

---

## Downloading the Dataset

```bash
!git clone https://github.com/laxmimerit/IMDB-Movie-Reviews-Large-Dataset-50k.git
```

```python
# Loading the train dataset
data_train = pd.read_excel('IMDB-Movie-Reviews-Large-Dataset-50k/train.xlsx', dtype=str)

# Loading the test dataset
data_test = pd.read_excel('IMDB-Movie-Reviews-Large-Dataset-50k/test.xlsx', dtype=str)

# Dimension of the dataset
print("Size of train dataset: ", data_train.shape)
print("Size of test dataset: ", data_test.shape)
```

**Output**:
```
Size of train dataset:  (25000, 2)
Size of test dataset:  (25000, 2)
```

---

## Observations

- Both train and test datasets have **25,000 rows** and **2 columns**.

```python
# Printing last rows of train dataset
data_train.tail()
```

```python
# Printing head rows of test dataset
data_test.head()
```

---

## Splitting Data into Train and Test

```python
(X_train, y_train), (X_test, y_test), preproc = text.texts_from_df(
    train_df=data_train,
    text_column='Reviews',
    label_columns='Sentiment',
    val_df=data_test,
    maxlen=500,
    preprocess_mode='bert'
)
```

---

## Downloading Pretrained BERT Model

```
downloading pretrained BERT model (uncased_L-12_H-768_A-12.zip)...
[██████████████████████████████████████████████████]
extracting pretrained BERT model...
done.

**Source**: https://kgptalkie.com/sentiment-classification-using-bert

cleanup downloaded zip...
done.

**Source**: https://kgptalkie.com/sentiment-classification-using-bert
```

---

## Preprocessing

```
preprocessing train...
language: en
Is Multi-Label? False
preprocessing test...
language: en
```

**Observation**:
- The language is detected as **English**.
- This is **not a multi-label classification**.

---

## Building the Model

```python
model = text.text_classifier(name='bert', train_data=(X_train, y_train), preproc=preproc)
```

```python
learner = ktrain.get_learner(
    model=model,
    train_data=(X_train, y_train),
    val_data=(X_test, y_test),
    batch_size=6
)
```

```python
learner.fit_onecycle(lr=2e-5, epochs=1)
```

---

## Predicting

```python
predictor = ktrain.get_predictor(learner.model, preproc)
predictor.save('/content/drive/My Drive/bert')

# Sample dataset to test on
data = [
    'this movie was horrible, the plot was really boring. acting was okay',
    'the fild is really sucked. there is not plot and acting was bad',
    'what a beautiful movie. great plot. acting was good. will see it again'
]

predictor.predict(data)
```

**Output**:
```
['neg', 'neg', 'pos']
```

---

## Prediction with Probabilities

```python
predictor.predict(data, return_proba=True)
```

**Output**:
```
array([[0.99797565, 0.00202436],
       [0.99606663, 0.00393336],
       [0.00292433, 0.9970757 ]], dtype=float32)
```

---

## Classes Available

```python
predictor.get_classes()
```

**Output**:
```
['neg', 'pos']
```

---

## Saving the Model

```python
predictor.save('/content/drive/My Drive/bert')
!zip -r /content/bert.zip /content/bert
```

---

## Loading the Model

```python
predictor_load = ktrain.load_predictor('/content/bert')
predictor_load.predict(data)
```

**Output**:
```
['neg', 'neg', 'pos']
```

---

## Summary

1. We have loaded the dataset and processed it using **pandas**.
2. We have used **pre-trained BERT** model weights on our dataset using the **kTrain** library.
3. We have found the best learning parameter and used it to fit the model.
4. Finally, using that model, we have predicted the output.

---

## Note

For better performance, consider using **distilBERT**, a variant of BERT.