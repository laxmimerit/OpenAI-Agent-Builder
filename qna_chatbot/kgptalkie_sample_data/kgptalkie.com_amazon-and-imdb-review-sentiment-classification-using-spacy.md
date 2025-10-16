https://kgptalkie.com/amazon-and-imdb-review-sentiment-classification-using-spacy

# Amazon and IMDB Review Sentiment Classification using SpaCy

**Published by** georgiannacambel **on** 12 September 2020

## Sentiment Classification using SpaCy

### What is NLP?
Natural Language Processing (NLP) is the field of Artificial Intelligence concerned with the processing and understanding of human language. Since its inception during the 1950s, machine understanding of language has played a pivotal role in translation, topic modeling, document indexing, information retrieval, and extraction.

### Some Applications of NLP
- Text Classification
- Spam Filters
- Voice text messaging
- Sentiment analysis
- Spell or grammar check
- Chat bot
- Search Suggestion
- Search Autocorrect
- Automatic Review Analysis system
- Machine translation

## spaCy installation
You can run the following commands:-
```python
!pip install -U spacy
!pip install -U spacy-lookups-data
!python -m spacy download en_core_web_sm
```

## Scikit-learn installation
You can run the following command:-
```python
!pip install scikit-learn
```

## Data Cleaning Options
- Case Normalization
- Removing Stop Words
- Removing Punctuations or Special Symbols
- Lemmatization or Stemming
- Parts of Speech Tagging
- Entity Detection
- Bag of Words
- TF-IDF

### Bag of Words – The Simplest Word Embedding Technique
This is one of the simplest methods of embedding words into numerical vectors. It is not often used in practice due to its oversimplification of language, but often the first embedding technique to be taught in the classroom setting. Whenever we apply any algorithm in NLP, it works on numbers. We cannot directly feed our text into that algorithm. Hence, Bag of Words model is used to preprocess the text by converting it into a bag of words, which keeps a count of the total occurrences of unique words.

A bag-of-words is a representation of text that describes the occurrence of words within a document. It involves two things:
- A vocabulary of known words.
- A measure of the presence of known words.

It is called a “bag” of words, because any information about the order or structure of words in the document is discarded. The model is only concerned with whether known words occur in the document, not where in the document.

```python
doc1 = "I am high"
doc2 = "Yes I am high"
doc3 = "I am kidding"
```

### Bag of Words and Tf-idf
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html

In information retrieval, tf–idf or TFIDF, short for term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. This is done by multiplying two metrics: how many times a word appears in a document, and the inverse document frequency of the word across a set of documents.

## Pipeline in SpaCy
When you call `nlp` on a text, spaCy first tokenizes the text to produce a `Doc` object. The `Doc` is then processed in several different steps – this is also referred to as the processing pipeline.

The pipeline used by the default models consists of a tagger, a parser and an entity recognizer. Each pipeline component returns the processed `Doc`, which is then passed on to the next component.

## Datasets
You can get all the datasets used in this notebook from [here](https://kgptalkie.com/amazon-and-imdb-review-sentiment-classification-using-spacy).

Watch Full Video here: [Let’s Get Started](https://kgptalkie.com/amazon-and-imdb-review-sentiment-classification-using-spacy)

## Code Implementation

### Importing Libraries
```python
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')
text = "Apple, This is first sentence. and Google this is another one. here 3rd one is"
doc = nlp(text)
```

### Tokens in `doc`
```python
for token in doc:
    print(token)
```

### Creating Pipeline
```python
sent = nlp.create_pipe('sentencizer')
nlp.add_pipe(sent, before='parser')
doc = nlp(text)
for sent in doc.sents:
    print(sent)
```

### Stop Words
```python
from spacy.lang.en.stop_words import STOP_WORDS
stopwords = list(STOP_WORDS)
print(stopwords)
```

**Source:** https://kgptalkie.com/amazon-and-imdb-review-sentiment-classification-using-spacy

### Lemmatization
```python
doc = nlp('run runs running runner')
for lem in doc:
    print(lem.text, lem.lemma_)
```

### POS Tagging
```python
doc = nlp('All is well at your end!')
for token in doc:
    print(token.text, token.pos_)
```

### Entity Detection
```python
doc = nlp("New York City on Tuesday declared a public health emergency and ordered mandatory measles vaccinations amid an outbreak, becoming the latest national flash point over refusals to inoculate against dangerous diseases. At least 285 people have contracted measles in the city since September, mostly in Brooklyn’s Williamsburg neighborhood. The order covers four Zip codes there, Mayor Bill de Blasio (D) said Tuesday. The mandate orders all unvaccinated people in the area, including a concentration of Orthodox Jews, to receive inoculations, including for children as young as 6 months old. Anyone who resists could be fined up to $1,000.")
displacy.render(doc, style = 'ent')
```

## Text Classification

### Importing Libraries
```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

### Loading Datasets
```python
data_yelp = pd.read_csv('datasets/yelp_labelled.txt', sep='\t', header=None)
data_yelp.columns = ['Review', 'Sentiment']
data_yelp.head()
```

```python
data_amazon = pd.read_csv('datasets/amazon_cells_labelled.txt', sep='\t', header=None)
data_amazon.columns = ['Review', 'Sentiment']
data_amazon.head()
```

```python
data_imdb = pd.read_csv('datasets/imdb_labelled.txt', sep='\t', header=None)
data_imdb.columns = ['Review', 'Sentiment']
data_imdb.head()
```

### Combining Datasets
```python
data = data_yelp.append([data_amazon, data_imdb], ignore_index=True)
data.shape
```

### Data Cleaning
```python
import string
punct = string.punctuation
punct
```

```python
def text_data_cleaning(sentence):
    doc = nlp(sentence)
    tokens = []
    for token in doc:
        if token.lemma_ != "-PRON-":
            temp = token.lemma_.lower().strip()
        else:
            temp = token.lower_
        tokens.append(temp)
    cleaned_tokens = []
    for token in tokens:
        if token not in stopwords and token not in punct:
            cleaned_tokens.append(token)
    return cleaned_tokens
```

### Vectorization and Model Training
```python
tfidf = TfidfVectorizer(tokenizer=text_data_cleaning)
classifier = LinearSVC()
X = data['Review']
y = data['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

```python
clf = Pipeline([('tfidf', tfidf), ('clf', classifier)])
clf.fit(X_train, y_train)
```

### Model Evaluation
```python
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

**Source:** https://kgptalkie.com/amazon-and-imdb-review-sentiment-classification-using-spacy

```python
print(confusion_matrix(y_test, y_pred))
```

### Predicting New Sentences
```python
clf.predict(['Wow, this is amazing lesson'])
clf.predict(['Wow, this sucks'])
clf.predict(['Worth of watching it. Please like it'])
clf.predict(['Loved it. Amazing'])
```

**Source:** https://kgptalkie.com/amazon-and-imdb-review-sentiment-classification-using-spacy

In this blog we saw some features of SpaCy. Then we went ahead and performed sentiment analysis by loading the data, pre-processing it and then training our model. We used tf-idf vectorizer and Linear SVC to train the model. We got an accuracy of 78%.