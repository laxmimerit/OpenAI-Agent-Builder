https://kgptalkie.com/text-summarization-using-nlp

# Text Summarization using NLP

**Source:** https://kgptalkie.com/text-summarization-using-nlp

Published by  
georgiannacambel  
on  
4 September 2020

## Extractive Text Summarization

### What is text summarization?

Text summarization is the process of creating a short, accurate, and fluent summary of a longer text document. It is the process of distilling the most important information from a source text. Automatic text summarization is a common problem in [machine learning](https://en.wikipedia.org/wiki/Machine_learning) and natural language processing (NLP). Automatic text summarization methods are greatly needed to address the ever-growing amount of text data available online to both better help discover relevant information and to consume relevant information faster.

### Why automatic text summarization?

- Summaries reduce reading time.
- While researching using various documents, summaries make the selection process easier.
- Automatic summarization improves the effectiveness of indexing.
- Automatic summarization algorithms are less biased than human summarizers.
- Personalized summaries are useful in question-answering systems as they provide personalized information.
- Using automatic or semi-automatic summarization systems enables commercial abstract services to – increase the number of text documents they are able to process.

### Type of summarization

- **Extractive summarization**: Selects important sentences, paragraphs, etc., from the original document and concatenates them into a shorter form.
- **Abstractive summarization**: Understands the main concepts in a document and expresses those concepts in clear natural language.
- **Domain-specific summarization**: Utilizes domain-specific knowledge (e.g., medical ontologies).
- **Generic summarization**: Focuses on obtaining a generic summary of a collection of documents, images, videos, etc.
- **Query-based summarization**: Summarizes objects specific to a query.
- **Multi-document summarization**: Extracts information from multiple texts about the same topic.
- **Single-document summarization**: Generates a summary from a single source document.

## How to do text summarization

### Text cleaning
### Sentence Tokenization
### Word tokenization
### Word-frequency table
### Summarization

### Text
This is the piece of text which we will be using in this project. We will perform extractive summarization of this text.

```python
text = """
There are broadly two types of extractive summarization tasks depending on what the summarization program focuses on. The first is generic summarization, which focuses on obtaining a generic summary or abstract of the collection (whether documents, or sets of images, or videos, news stories etc.). The second is query relevant summarization, sometimes called query-based summarization, which summarizes objects specific to a query. Summarization systems are able to create both query relevant text summaries and generic machine-generated summaries depending on what the user needs.
An example of a summarization problem is document summarization, which attempts to automatically produce an abstract from a given document. Sometimes one might be interested in generating a summary from a single source document, while others can use multiple source documents (for example, a cluster of articles on the same topic). This problem is called multi-document summarization. A related application is summarizing news articles. Imagine a system, which automatically pulls together news articles on a given topic (from the web), and concisely represents the latest news as a summary.
Image collection summarization is another application example of automatic summarization. It consists in selecting a representative set of images from a larger set of images.[3] A summary in this context is useful to show the most representative images of results in an image collection exploration system. Video summarization is a related domain, where the system automatically creates a trailer of a long video. This also has applications in consumer or personal videos, where one might want to skip the boring or repetitive actions. Similarly, in surveillance videos, one would want to extract important and suspicious activity, while ignoring all the boring and redundant frames captured.
"""
```

Let’s Get Started with **SpaCy**

You can install SpaCy using the following commands:  
```bash
!pip install -U spacy
!python -m spacy download en_core_web_sm
```

We will import all the necessary libraries.

```python
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
```

Here we will create a list of **stopwords**.

```python
stopwords = list(STOP_WORDS)
```

```python
nlp = spacy.load('en_core_web_sm')
```

Calling the `nlp` object on a string of text will return a processed `Doc`.

```python
doc = nlp(text)
```

Each `Doc` consists of individual tokens, and we can iterate over them. Now we will make a list of tokens called `tokens`.

```python
tokens = [token.text for token in doc]
print(tokens)
```

Output:
```
['\n', 'There', 'are', 'broadly', 'two', 'types', 'of', 'extractive', 'summarization', 'tasks', 'depending', 'on', 'what', 'the', 'summarization', 'program', 'focuses', 'on', '.', 'The', 'first', 'is', 'generic', 'summarization', ',', 'which', 'focuses', 'on', 'obtaining', 'a', 'generic', 'summary', 'or', 'abstract', 'of', 'the', 'collection', '(', 'whether', 'documents', ',', 'or', 'sets', 'of', 'images', ',', 'or', 'videos', ',', 'news', 'stories', 'etc', '.', ')', '.', 'The', 'second', 'is', 'query', 'relevant', 'summarization', ',', 'sometimes', 'called', 'query', '-', 'based', 'summarization', ',', 'which', 'summarizes', 'objects', 'specific', 'to', 'a', 'query', '.', 'Summarization', 'systems', 'are', 'able', 'to', 'create', 'both', 'query', 'relevant', 'text', 'summaries', 'and', 'generic', 'machine', '-', 'generated', 'summaries', 'depending', 'on', 'what', 'the', 'user', 'needs', '.', '\n', 'An', 'example', 'of', 'a', 'summarization', 'problem', 'is', 'document', 'summarization', ',', 'which', 'attempts', 'to', 'automatically', 'produce', 'an', 'abstract', 'from', 'a', 'given', 'document', '.', 'Sometimes', 'one', 'might', 'be', 'interested', 'in', 'generating', 'a', 'summary', 'from', 'a', 'single', 'source', 'document', ',', 'while', 'others', 'can', 'use', 'multiple', 'source', 'documents', '(', 'for', 'example', ',', 'a', 'cluster', 'of', 'articles', 'on', 'the', 'same', 'topic', ')', '.', 'This', 'problem', 'is', 'called', 'multi', '-', 'document', 'summarization', '.', 'A', 'related', 'application', 'is', 'summarizing', 'news', 'articles', '.', 'Imagine', 'a', 'system', ',', 'which', 'automatically', 'pulls', 'together', 'news', 'articles', 'on', 'a', 'given', 'topic', '(', 'from', 'the', 'web', ')', ',', 'and', 'concisely', 'represents', 'the', 'latest', 'news', 'as', 'a', 'summary', '.', '\n', 'Image', 'collection', 'summarization', 'is', 'another', 'application', 'example', 'of', 'automatic', 'summarization', '.', 'It', 'consists', 'in', 'selecting', 'a', 'representative', 'set', 'of', 'images', 'from', 'a', 'larger', 'set', 'of', 'images.[3', ']', 'A', 'summary', 'in', 'this', 'context', 'is', 'useful', 'to', 'show', 'the', 'most', 'representative', 'images', 'of', 'results', 'in', 'an', 'image', 'collection', 'exploration', 'system', '.', 'Video', 'summarization', 'is', 'a', 'related', 'domain', ',', 'where', 'the', 'system', 'automatically', 'creates', 'a', 'trailer', 'of', 'a', 'long', 'video', '.', 'This', 'also', 'has', 'applications', 'in', 'consumer', 'or', 'personal', 'videos', ',', 'where', 'one', 'might', 'want', 'to', 'skip', 'the', 'boring', 'or', 'repetitive', 'actions', '.', 'Similarly', ',', 'in', 'surveillance', 'videos', ',', 'one', 'would', 'want', 'to', 'extract', 'important', 'and', 'suspicious', 'activity', ',', 'while', 'ignoring', 'all', 'the', 'boring', 'and', 'redundant', 'frames', 'captured', '.', '\n']
```

We can see that all the punctuation marks and special characters are included in the tokens. Now we will remove them.

```python
punctuation = punctuation + '\n'
print(punctuation)
```

Output:
```
!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~\n
```

Now we will make the **word frequency table**. It will contain the number of occurrences of all the distinct words in the text which are not punctuations or stop words.

```python
word_frequencies = {}
for word in doc:
    if word.text.lower() not in stopwords:
        if word.text.lower() not in punctuation:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1
print(word_frequencies)
```

Output:
```python
{'broadly': 1, 'types': 1, 'extractive': 1, 'summarization': 11, 'tasks': 1, 'depending': 2, 'program': 1, 'focuses': 2, 'generic': 3, 'obtaining': 1, 'summary': 4, 'abstract': 2, 'collection': 3, 'documents': 2, 'sets': 1, 'images': 3, 'videos': 3, 'news': 4, 'stories': 1, 'etc': 1, 'second': 1, 'query': 4, 'relevant': 2, 'called': 2, 'based': 1, 'summarizes': 1, 'objects': 1, 'specific': 1, 'Summarization': 1, 'systems': 1, 'able': 1, 'create': 1, 'text': 1, 'summaries': 2, 'machine': 1, 'generated': 1, 'user': 1, 'needs': 1, 'example': 3, 'problem': 2, 'document': 4, 'attempts': 1, 'automatically': 3, 'produce': 1, 'given': 2, 'interested': 1, 'generating': 1, 'single': 1, 'source': 2, 'use': 1, 'multiple': 1, 'cluster': 1, 'articles': 3, 'topic': 2, 'multi': 1, 'related': 2, 'application': 2, 'summarizing': 1, 'Imagine': 1, 'system': 3, 'pulls': 1, 'web': 1, 'concisely': 1, 'represents': 1, 'latest': 1, 'Image': 1, 'automatic': 1, 'consists': 1, 'selecting': 1, 'representative': 2, 'set': 2, 'larger': 1, 'images.[3': 1, 'context': 1, 'useful': 1, 'results': 1, 'image': 1, 'exploration': 1, 'Video': 1, 'domain': 1, 'creates': 1, 'trailer': 1, 'long': 1, 'video': 1, 'applications': 1, 'consumer': 1, 'personal': 1, 'want': 2, 'skip': 1, 'boring': 2, 'repetitive': 1, 'actions': 1, 'Similarly': 1, 'surveillance': 1, 'extract': 1, 'important': 1, 'suspicious': 1, 'activity': 1, 'ignoring': 1, 'redundant': 1, 'frames': 1, 'captured': 1}
```

Now we will get the **max_frequency**.

```python
max_frequency = max(word_frequencies.values())
print(max_frequency)
```

Output:
```
11
```

We will divide each frequency value in `word_frequencies` with the `max_frequency` to normalize the frequencies.

```python
for word in word_frequencies.keys():
    word_frequencies[word] = word_frequencies[word]/max_frequency
print(word_frequencies)
```

**Source:** https://kgptalkie.com/text-summarization-using-nlp

Now we will do **sentence tokenization**. The entire text is divided into sentences.

```python
sentence_tokens = [sent for sent in doc.sents]
print(sentence_tokens)
```

Output:
```python
[
There are broadly two types of extractive summarization tasks depending on what the summarization program focuses on., The first is generic summarization, which focuses on obtaining a generic summary or abstract of the collection (whether documents, or sets of images, or videos, news stories etc.)., The second is query relevant summarization, sometimes called query-based summarization, which summarizes objects specific to a query., Summarization systems are able to create both query relevant text summaries and generic machine-generated summaries depending on what the user needs.
, An example of a summarization problem is document summarization, which attempts to automatically produce an abstract from a given document., Sometimes one might be interested in generating a summary from a single source document, while others can use multiple source documents (for example, a cluster of articles on the same topic)., This problem is called multi-document summarization., A related application is summarizing news articles., Imagine a system, which automatically pulls together news articles on a given topic (from the web), and concisely represents the latest news as a summary.
, Image collection summarization is another application example of automatic summarization., It consists in selecting a representative set of images from a larger set of images.[3], A summary in this context is useful to show the most representative images of results in an image collection exploration system., Video summarization is a related domain, where the system automatically creates a trailer of a long video., This also has applications in consumer or personal videos, where one might want to skip the boring or repetitive actions., Similarly, in surveillance videos, one would want to extract important and suspicious activity, while ignoring all the boring and redundant frames captured.
]
```

Now we will calculate the sentence scores. The sentence score for a particular sentence is the sum of the normalized frequencies of the words in that sentence.

```python
sentence_scores = {}
for sent in sentence_tokens:
    for word in sent:
        if word.text.lower() in word_frequencies.keys():
            if sent not in sentence_scores.keys():
                sentence_scores[sent] = word_frequencies[word.text.lower()]
            else:
                sentence_scores[sent] += word_frequencies[word.text.lower()]
sentence_scores
```

Output:
```python
{
 There are broadly two types of extractive summarization tasks depending on what the summarization program focuses on.: 2.818181818181818,
 The first is generic summarization, which focuses on obtaining a generic summary or abstract of the collection (whether documents, or sets of images, or videos, news stories etc.).: 3.9999999999999987,
 The second is query relevant summarization, sometimes called query-based summar意图: 3.909090909090909,
 Summarization systems are able to create both query relevant text summaries and generic machine-generated summaries depending on what the user needs.: 3.09090909090909,
 An example of a summarization problem is document summarization, which attempts to automatically produce an abstract from a given document.: 3.9999999999999996,
 Sometimes one might be interested in generating a summary from a single source document, while others can use multiple source documents (for example, a cluster of articles on the same topic).: 2.545454545454545,
 This problem is called multi-document summarization.: 1.8181818181818183,
 A related application is summarizing news articles.: 1.0909090909090908,
 Imagine a system, which automatically pulls together news articles on a given topic (from the web), and concisely represents the latest news as a summary.: 2.727272727272727,
 Image collection summarization is another application example of automatic summarization.: 2.909090909090909,
 It consists in selecting a representative set of images from a larger set of images.[3]: 1.1818181818181817,
 A summary in this context is useful to show the most representative images of results in an image collection exploration system.: 1.818181818181818,
 Video summarization is a related domain, where the system automatically creates a trailer of a long video.: 2.2727272727272725,
 This also has applications in consumer or personal videos, where one might want to skip the boring or repetitive actions.: 1.1818181818181817,
 Similarly, in surveillance videos, one would want to extract important and suspicious activity, while ignoring all the boring and redundant frames captured.: 1.4545454545454544}
```

Now we are going to select 30% of the sentences having the largest scores. For this, we are going to import `nlargest` from `heapq`.

```python
from heapq import nlargest
```

We want the length of the summary to be 30% of the original length, which is 4. Hence, the summary will have 4 sentences.

```python
select_length = int(len(sentence_tokens)*0.3)
print(select_length)
```

Output:
```
4
```

```python
summary = nlargest(select_length, sentence_scores, key = sentence_scores.get)
print(summary)
```

Output:
```python
[An example of a summarization problem is document summarization, which attempts to automatically produce an abstract from a given document.,
 The first is generic summarization, which focuses on obtaining a generic summary or abstract of the collection (whether documents, or sets of images, or videos, news stories etc.).,
 The second is query relevant summarization, sometimes called query-based summarization, which summarizes objects specific to a query.,
 Summarization systems are able to create both query relevant text summaries and generic machine-generated summaries depending on what the user needs.]
```

Now we will combine this sentence together and make the final **string** which contains the summary.

```python
final_summary = [word.text for word in summary]
summary = ' '.join(final_summary)
```

Now we will display the original text, the summary of the text, and the lengths of the original text and the generated summary.

```python
print(text)
print(summary)
print(len(text))
print(len(summary))
```

**Source:** https://kgptalkie.com/text-summarization-using-nlp