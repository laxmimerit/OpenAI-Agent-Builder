# [DistilBERT – Smaller, faster, cheaper, lighter and ofcourse Distilled!](https://kgptalkie.com/distilbert-smaller-faster-cheaper-lighter-and-ofcourse-distilled)

**Source:** [https://kgptalkie.com/distilbert-smaller-faster-cheaper-lighter-and-ofcourse-distilled](https://kgptalkie.com/distilbert-smaller-faster-cheaper-lighter-and-ofcourse-distilled)

## Published by
berryedelson  
25 August 2020

---

## Sentiment Classification Using DistilBERT

### Problem Statement
We will use the **IMDB Movie Reviews Dataset**, where based on the given review we have to classify the sentiments of that particular review like **positive** or **negative**.

---

## The Motivational BERT

**BERT** became an essential ingredient of many NLP deep learning pipelines. It is considered a milestone in NLP, as **ResNet** is in the computer vision field.

Google published an article **"Understanding searches better than ever before"** and positioned **BERT** as one of its most important updates to the searching algorithms in recent years. **BERT** is a language representation model with impressive accuracy for many NLP tasks.

**BERT** is designed to pre-train deep bidirectional representations from the unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained **BERT** model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

---

## What is **DistilBERT**?

### Distillation
Another interesting model compression method is **distillation** — a technique that transfers the knowledge of a large "teacher" network to a smaller "student" network. The "student" network is trained to mimic the behaviors of the "teacher" network.

A version of this strategy has already been pioneered by **Rich Caruana** and his collaborators. In their important paper, they demonstrate convincingly that the knowledge acquired by a large ensemble of models can be transferred to a single small model. **Geoffrey Hinton** et al. showed this technique can be applied to neural networks in their paper called **"Distilling the Knowledge in a Neural Network"**.

Since then, this approach was applied to different neural networks, and you probably heard of a **BERT distillation** called **DistilBERT** by **HuggingFace**.

Finally, October 2nd, a paper on **DistilBERT** called **"DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"** emerged and was submitted at **NeurIPS 2019**.

**DistilBERT** is a smaller language model, trained from the supervision of **BERT** in which authors removed the token-type embeddings and the pooler (used for the next sentence classification task) and kept the rest of the architecture identical while reducing the numbers of layers by a factor of two.

**DistilBERT** is a small, fast, cheap, and light Transformer model trained by distilling **BERT base**. It has **40% less parameters** than **bert-base-uncased**, runs **60% faster** while preserving over **95% of BERT’s performances** as measured on the **GLUE language understanding benchmark**.

In terms of inference time, **DistilBERT** is more than **60% faster** and smaller than **BERT** and **120% faster** and smaller than **ELMo+BiLSTM**.

---

### Why **DistilBERT**?
- Accurate as much as Original **BERT** Model
- **60% faster**
- **40% fewer parameters**
- It can run on **CPU**

---

## Additional Reading

- [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108)
- [Video Lecture: BERT NLP Tutorial 1- Introduction | BERT Machine Learning | KGP Talkie](https://www.youtube.com/watch?v=UosUDxYs96A)
- [Ref BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Understanding searches better than ever before](https://www.blog.google/products/search/search-language-understanding-bert/)
- [Good Resource to Read More About the BERT](http://jalammar.github.io/illustrated-bert/)
- [Visual Guide to Using BERT](http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)

---

## What is **ktrain**?

**ktrain** is a library to help build, train, debug, and deploy neural networks in the deep learning software framework, **Keras**.

**ktrain** uses **tf.keras** in **TensorFlow** instead of standalone **Keras**. Inspired by the **fastai** library, with only a few lines of code, **ktrain** allows you to easily:

- Estimate an optimal learning rate for your model given your data using a learning rate finder
- Employ learning rate schedules such as the triangular learning rate policy, 1cycle policy, and SGDR to more effectively train your model
- Employ fast and easy-to-use pre-canned models for both text classification (e.g., NBSVM, fastText, GRU with pre-trained word embeddings) and image classification (e.g., ResNet, Wide Residual Networks, Inception)
- Load and preprocess text and image data from a variety of formats
- Inspect data points that were misclassified to help improve your model
- Leverage a simple prediction API for saving and deploying both models and data-preprocessing steps to make predictions on new raw data

**ktrain GitHub:** [https://github.com/amaiya/ktrain](https://github.com/amaiya/ktrain)

---

## Notebook Setup

```bash
!pip install ktrain
```

```bash
!git clone https://github.com/laxmimerit/IMDB-Movie-Reviews-Large-Dataset-50k.git
```

---

## Importing Libraries

```python
import pandas as pd
import numpy as np
import ktrain
from ktrain import text
import tensorflow as tf
```

---

## Loading Dataset

```python
# loading the training and testing dataset
data_test = pd.read_excel('/content/IMDB-Movie-Reviews-Large-Dataset-50k/test.xlsx', dtype= str)
data_train = pd.read_excel('/content/IMDB-Movie-Reviews-Large-Dataset-50k/train.xlsx', dtype = str)

# printing the five sample datapoints
data_train.sample(5)
```

---

## Text Classifiers

```python
# printing the available text classifiers models
text.print_text_classifiers()
```

---

## Preprocessing

```python
(train, val, preproc) = text.texts_from_df(
    train_df=data_train, 
    text_column='Reviews', 
    label_columns='Sentiment',
    val_df=data_test,
    maxlen=400,
    preprocess_mode='distilbert'
)
```

---

## Model Definition

```python
model = text.text_classifier(name='distilbert', train_data=train, preproc=preproc)
```

---

## Training

```python
learner = ktrain.get_learner(
    model=model,
    train_data=train,
    val_data=val,
    batch_size=6
)

learner.fit_onecycle(lr=2e-5, epochs=2)
```

---

## Prediction

```python
predictor = ktrain.get_predictor(learner.model, preproc)
predictor.save('/content/drive/My Drive/distilbert')

data = [
    'this movie was really bad. acting was also bad. I will not watch again',
    'the movie was really great. I will see it again',
    'another great movie. must watch to everyone'
]

predictor.predict(data)
```

---

## Results

```python
['neg', 'pos', 'pos']
```

**Interpretation of above results:**

- 'this movie was really bad. acting was also bad. I will not watch again' – **neg**
- 'the movie was really great. I will see it again' – **pos**
- 'another great movie. must watch to everyone' – **pos**

```python
predictor.get_classes()
['neg', 'pos']
```

```python
predictor.predict(data, return_proba=True)
array([[0.9944576 , 0.00554235],
       [0.00516187, 0.99483806],
       [0.00479033, 0.99520963]], dtype=float32)
```

---

## Summary

1. We have loaded the pre-loaded dataset and processed it using **pandas** dataframe.
2. Thereafter, we have used pre-trained model weights of **distilBERT** on our dataset using **kTrain** library.
3. Then, we have found the best learning parameter and using that we have fit the model.
4. Finally, using that model we have predicted our output.

**Source:** [https://kgptalkie.com/distilbert-smaller-faster-cheaper-lighter-and-ofcourse-distilled](https://kgptalkie.com/distilbert-smaller-faster-cheaper-lighter-and-ofcourse-distilled)