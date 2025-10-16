**Source:** https://kgptalkie.com/spacy-introduction-for-nlp-linguistic-features-extraction

# SpaCy Introduction for NLP | Linguistic Features Extraction

**Published by:** crystlefroggatt  
**Date:** 28 August 2020

## Getting Started with spaCy

This tutorial is a crisp and effective introduction to spaCy and the various NLP linguistic features it offers. We will perform several NLP-related tasks, such as:

- Tokenization
- Part-of-speech tagging
- Named entity recognition
- Dependency parsing
- Visualization using displaCy

### What is spaCy?

spaCy is a free, open-source library for advanced Natural Language Processing (NLP) in [Python](https://www.python.org/). spaCy is designed specifically for production use and helps you build applications that process and understand large volumes of text. It’s written in [Cython](https://cython.org/) and is designed to build:

- Information extraction systems
- Natural language understanding systems
- Pre-process text for deep learning

---

## Linguistic Features in spaCy

Processing raw text intelligently is difficult: most words are rare, and it’s common for words that look completely different to mean almost the same thing.

That’s exactly what spaCy is designed to do: you put in raw text, and get back a `Doc` object, that comes with a variety of Linguistic annotations.

spaCy acts as a one-stop-shop for various tasks used in NLP projects, such as:

- Tokenization
- Lemmatisation
- Part-of-speech (POS) tagging
- Name entity recognition
- Dependency parsing
- Sentence segmentation
- Word-to-vector transformations
- Other cleaning and normalization text methods

---

## Setup

Install spaCy and the English model using the following commands:

```bash
!pip install -U spacy
!pip install -U spacy-lookups-data
!python -m spacy download en_core_web_sm
```

Once installed, load the model:

```python
import spacy
nlp = spacy.load('en_core_web_sm')
```

> **Note:** The default model for the English language is `en_core_web_sm`.

---

## Tokenization

**Tokenization** is the task of splitting a text into meaningful segments called **tokens**.

Example:

```python
doc = nlp("Apple isn't looking at buyig U.K. startup for $1 billion")
for token in doc:
    print(token.text)
```

Output:

```
Apple
is
n't
looking
at
buyig
U.K.
startup
for
$
1
billion
```

---

## Lemmatization

**Lemmatization** is the method of reducing the word to its base form, or origin form. This reduced form or root word is called a **lemma**.

Example:

```python
for token in doc:
    print(token.text, token.lemma_)
```

Output:

```
Apple Apple
is be
n't not
looking look
at at
buyig buyig
U.K. U.K.
startup startup
for for
$ $
1 1
billion billion
```

---

## Part-of-Speech Tagging

**Part of speech tagging** is the process of assigning a **POS** tag to each token depending on its usage in the sentence.

Example:

```python
for token in doc:
    print(f'{token.text:{15}} {token.lemma_:{15}} {token.pos_:{10}} {token.is_stop}')
```

Output:

```
Apple           Apple           PROPN      False
is              be              AUX        True
n't             not             PART       True
looking         look            VERB       False
at              at              ADP        True
buyig           buyig           NOUN       False
U.K.            U.K.            PROPN      False
startup         startup         NOUN       False
for             for             ADP        True
$               $               SYM        False
1               1               NUM        False
billion         billion         NUM        False
```

---

## Dependency Parsing

**Dependency parsing** is the process of extracting the dependency parse of a sentence to represent its grammatical structure.

Example:

```python
for chunk in doc.noun_chunks:
    print(f'{chunk.text:{30}} {chunk.root.text:{15}} {chunk.root.dep_}')
```

Output:

```
Apple                          Apple           nsubj
buyig U.K. startup             startup         pobj
```

---

## Named Entity Recognition

**Named Entity Recognition (NER)** is the process of locating named entities in unstructured text and then classifying them into pre-defined categories, such as:

- Person names
- Organizations
- Locations
- Monetary values
- Percentages
- Time expressions

Example:

```python
for ent in doc.ents:
    print(ent.text, ent.label_)
```

Output:

```
Apple ORG
U.K. GPE
$1 billion MONEY
```

---

## Sentence Segmentation

**Sentence Segmentation** is the process of locating the start and end of sentences in a given text.

Example:

```python
doc = nlp("Welcome to KGP Talkie. Thanks for watching. Please like and subscribe")
for sent in doc.sents:
    print(sent)
```

Output:

```
Welcome to KGP Talkie.
Thanks for watching.
Please like and subscribe
```

> **Note:** If delimiters like `...` are used, custom rules may be needed to detect sentence boundaries.

Example of a custom rule:

```python
def set_rule(doc):
    for token in doc[:-1]:
        if token.text == '...':
            doc[token.i + 1].is_sent_start = True
    return doc

nlp.add_pipe(set_rule, before='parser')
```

---

## Visualization

spaCy comes with a built-in visualizer called **displaCy**. You can use it to visualize a dependency parse or named entities in a browser or a Jupyter notebook.

Example:

```python
from spacy import displacy
doc = nlp("Welcome to KGP Talkie...Thanks...Like and Subscribe!")
displacy.render(doc, style='dep')
```

For named entities:

```python
displacy.render(doc, style='ent')
```

---

## Conclusion

spaCy is a modern, reliable NLP framework that quickly became the standard for doing NLP with Python. Its main advantages are:

- **Speed**
- **Accuracy**
- **Extensibility**

We have gained insights into linguistic annotations like:

- Tokenization
- Lemmatisation
- Part-of-speech (POS) tagging
- Entity recognition
- Dependency parsing
- Sentence segmentation
- Visualization using displaCy

**Source:** https://kgptalkie.com/spacy-introduction-for-nlp-linguistic-features-extraction