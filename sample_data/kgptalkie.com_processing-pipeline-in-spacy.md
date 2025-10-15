# Processing Pipeline in SpaCy  
**Source:** https://kgptalkie.com/processing-pipeline-in-spacy  

## What is SpaCy?  
spaCy is a free, open-source library for advanced Natural Language Processing (NLP) in Python.  

If you’re working with a lot of text, you’ll eventually want to know more about it. For example:  
- What’s it about?  
- What do the words mean in context?  
- Who is doing what to whom?  
- What companies and products are mentioned?  
- Which texts are similar to each other?  

spaCy is designed specifically for production use and helps you build applications that process and “understand” large volumes of text. It can be used to build information extraction or natural language understanding systems, or to pre-process text for deep learning.  

Below are some of spaCy’s features and capabilities. Some of them refer to linguistic concepts, while others are related to more general **machine learning** functionality.  

---

## Pipeline in SpaCy  
When you call `nlp` on a text, spaCy first tokenizes the text to produce a `Doc` object. The `Doc` is then processed in several different steps – this is also referred to as the processing pipeline.  

The pipeline used by the default models consists of a **tagger**, a **parser**, and an **entity recognizer**. Each pipeline component returns the processed `Doc`, which is then passed on to the next component.  

---

## spaCy Installation  
You can run the following commands:  
```bash
!pip install -U spacy
!pip install -U spacy-lookups-data
!python -m spacy download en_core_web_sm
```  

**Source:** https://kgptalkie.com/processing-pipeline-in-spacy  

---

## Processing Text  
Here we have imported the necessary libraries:  
```python
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span
from spacy import displacy
```  

`spacy.load()` loads a model. When you call `nlp` on a text, spaCy will tokenize it and then call each component on the `Doc`, in order. It then returns the processed `Doc` that you can work with.  

```python
nlp = spacy.load("en_core_web_sm")
doc = nlp('This is raw text')
```  

When processing large volumes of text, the statistical models are usually more efficient if you let them work on batches of texts. spaCy’s `nlp.pipe` method takes an iterable of texts and yields processed `Doc` objects. The batching is done internally.  

```python
texts = ["This is raw text", "There is lots of text"]
docs = list(nlp.pipe(texts))
```  

---

## Tips for Efficient Processing  
- Process the texts as a stream using `nlp.pipe` and buffer them in batches, instead of one-by-one. This is usually much more efficient.  
- Only apply the pipeline components you need. Getting predictions from the model that you don’t actually need adds up and becomes very inefficient at scale. To prevent this, use the `disable` keyword argument to disable components you don’t need.  

In this example, we’re using `nlp.pipe` to process a (potentially very large) iterable of texts as a stream. Because we’re only accessing the named entities in `doc.ents` (set by the `ner` component), we’ll disable all other statistical components (the tagger and parser) during processing.  

```python
import spacy

texts = [
    "Net income was $9.4 million compared to the prior year of $2.7 million.",
    "Revenue exceeded twelve billion dollars, with a loss of $1b.",
]

nlp = spacy.load("en_core_web_sm")
docs = nlp.pipe(texts, disable=["tagger", "parser"])

for doc in docs:
    print([(ent.text, ent.label_) for ent in doc.ents])
    print()
```

**Output:**  
```
[('$9.4 million', 'MONEY'), ('the prior year', 'DATE'), ('$2.7 million', 'MONEY')]

[('twelve billion dollars', 'MONEY'), ('1b', 'MONEY')]
```

**Source:** https://kgptalkie.com/processing-pipeline-in-spacy  

---

## How Pipelines Work  
spaCy makes it very easy to create your own pipelines consisting of reusable components – this includes spaCy’s default tagger, parser, and entity recognizer, but also your own custom processing functions.  

A pipeline component can be added to an already existing `nlp` object, specified when initializing a `Language` class, or defined within a model package.  

When you load a model, spaCy first consults the model’s `meta.json`. The meta typically includes the model details, the ID of a language class, and an optional list of pipeline components. spaCy then does the following:  

1. Load the language class and data for the given ID via `get_lang_class` and initialize it.  
2. Iterate over the pipeline names and create each component using `create_pipe`, which looks them up in `Language.factories`.  
3. Add each pipeline component to the pipeline in order, using `add_pipe`.  
4. Make the model data available to the `Language` class by calling `from_disk` with the path to the model data directory.  

Example `meta.json` content:  
```json
{
  "lang": "en",
  "name": "core_web_sm",
  "description": "Example model for spaCy",
  "pipeline": ["tagger", "parser", "ner"]
}
```  

Fundamentally, a spaCy model consists of three components:  
- The weights (binary data loaded from a directory).  
- A pipeline of functions called in order.  
- Language data like tokenization rules and annotation schemes.  

---

## Disabling and Modifying Pipeline Components  
If you don’t need a particular component of the pipeline – for example, the `tagger` or the `parser` – you can disable loading it. This can sometimes make a big difference and improve loading speed.  

```python
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser"])
```  

In some cases, you do want to load all pipeline components and their weights, because you need them at different points in your application. However, if you only need a `Doc` object with named entities, there’s no need to run all pipeline components on it.  

```python
doc = nlp("Apple is buying a startup")
for ent in doc.ents:
    print(ent.text, ent.label_)
```

**Output:**  
```
Apple ORG
```  

Now we are disabling `ner` also. After disabling `ner`, we do not get any output.  

```python
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
doc = nlp("Apple is buying a startup")
for ent in doc.ents:
    print(ent.text, ent.label_)
```  

Now suppose we have a large document and we want to disable some pipeline components for a particular part of the text. Use the `with` block to disable components temporarily:  

```python
nlp = spacy.load('en_core_web_sm')

# 1. Use as a context manager
with nlp.disable_pipes("tagger", "parser"):
    doc = nlp("I won't be tagged and parsed")
doc = nlp("I will be tagged and parsed")
```  

Alternatively, `disable_pipes` returns an object that lets you call its `restore()` method to restore the disabled components when needed:  

```python
# 2. Restore manually
disabled = nlp.disable_pipes("ner")
doc = nlp("I won't have named entities")
disabled.restore()
```  

**Source:** https://kgptalkie.com/processing-pipeline-in-spacy