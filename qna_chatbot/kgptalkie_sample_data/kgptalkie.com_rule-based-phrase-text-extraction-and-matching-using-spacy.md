# Rule-Based Phrase Text Extraction and Matching Using SpaCy  
**Source:** https://kgptalkie.com/rule-based-phrase-text-extraction-and-matching-using-spacy  

---

## Published by  
georgiannacambel  
on  
9 September 2020  

---

## Text Extraction and Matching  

spaCy is a free, open-source library for advanced Natural Language Processing (NLP) in Python. If you’re working with a lot of text, you’ll eventually want to know more about it. For example:  
- What’s it about?  
- What do the words mean in context?  
- Who is doing what to whom?  
- What companies and products are mentioned?  
- Which texts are similar to each other?  

spaCy is designed specifically for production use and helps you build applications that process and “understand” large volumes of text. It can be used to build information extraction or natural language understanding systems, or to pre-process text for deep learning.  

### spaCy’s Features and Capabilities  
Some of spaCy’s features and capabilities include:  
- Linguistic concepts (e.g., tokenization, part-of-speech tagging)  
- General machine learning functionality  

---

## spaCy Installation  

You can run the following commands:  
```bash
!pip install -U spacy
!pip install -U spacy-lookups-data
!python -m spacy download en_core_web_sm
```

---

## spaCy Pipelining  

When you call `nlp` on a text, spaCy first tokenizes the text to produce a **Doc** object. The **Doc** is then processed in several different steps – this is also referred to as the processing pipeline. The pipeline used by the default models consists of:  
- A **tagger**  
- A **parser**  
- An **entity recognizer**  

Each pipeline component returns the processed **Doc**, which is then passed on to the next component.  

---

## Rule-Based Matching  

Compared to using regular expressions on raw text, spaCy’s rule-based matcher engines and components not only let you find the words and phrases you’re looking for – they also give you access to the tokens within the document and their relationships.  

This means you can:  
- Easily access and analyze the surrounding tokens  
- Merge spans into single tokens  
- Add entries to the named entities in `doc.ents`  

---

## Token-Based Matching  

spaCy features a rule-matching engine, the **Matcher**, that operates over tokens, similar to regular expressions.  

### Patterns  
The rules can refer to:  
- Token annotations (e.g., the token text or `tag_`)  
- Flags (e.g., `IS_PUNCT`)  

The rule matcher also lets you pass in a custom callback to act on matches – for example, to merge entities and apply custom labels.  

You can also associate patterns with entity IDs, to allow some basic entity linking or disambiguation. To match large terminology lists, you can use the **PhraseMatcher**, which accepts `Doc` objects as match patterns.  

---

## Adding Patterns  

Let’s say we want to enable spaCy to find a combination of three tokens:  
1. A token whose lowercase form matches “hello” (e.g., “Hello” or “HELLO”)  
2. A token whose `is_punct` flag is set to `True` (i.e., any punctuation)  
3. A token whose lowercase form matches “world” (e.g., “World” or “WORLD”)  

```python
[{"LOWER": "hello"}, {"IS_PUNCT": True}, {"LOWER": "world"}]
```

When writing patterns, keep in mind that each dictionary represents one token. If spaCy’s tokenization doesn’t match the tokens defined in a pattern, the pattern is not going to produce any results. When developing complex patterns, make sure to check examples against spaCy’s tokenization.  

---

## Code Example  

### Importing Libraries  
```python
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span
from spacy import displacy
```

### Loading the Model  
```python
nlp = spacy.load('en_core_web_sm')
doc = nlp('Hello World!')
```

### Token Matching  
```python
for token in doc:
    print(token)
```

**Output:**  
```
Hello
World
!
```

### Adding a Pattern  
```python
pattern = [{"LOWER": "hello", 'OP':'?'}, {"IS_PUNCT": True, 'OP':'?'}, {"LOWER": "world"}]
matcher = Matcher(nlp.vocab)
matcher.add('HelloWorld', None, pattern)
doc = nlp("Hello, world!")
matches = matcher(doc)
```

**Output:**  
```python
[(15578876784678163569, 0, 3), (15578876784678163569, 1, 3), (15578876784678163569, 2, 3)]
```

---

## Regular Expression  

In some cases, only matching tokens and token attributes isn’t enough – for example, you might want to match different spellings of a word, without having to add a new pattern for each spelling.  

### Example: Regex in spaCy  
```python
pattern = [{"TEXT": {"REGEX": "deff?in[ia]tely"}}]
```

### Matching POS Tags  
```python
pattern = [{"TAG": {"REGEX": "^V"}}]
```

---

## Visual Representation  

You can get a visual representation of Phrase Extraction by visiting this link:  
https://explosion.ai/demos/matcher  

---

## Wildcard Text Examples  

- Finding all words starting with `c` and having 2 characters after `c`:  
  ```python
  ['ct ', 'cal']
  ```

- Finding three-letter words with `a` as the middle character:  
  ```python
  ['cat', 'hat', 'wan', 'hat', ' an', 'cat']
  ```

- Strings ending with a number:  
  ```python
  ['3']
  ```

- Strings starting with a number:  
  ```python
  ['3']
  ```

- Excluding numbers:  
  ```python
  [' hi thanks for watching <']
  ```

- Extracting only numbers:  
  ```python
  ['33', '3']
  ```

- Finding words with hyphens:  
  ```python
  ['free-videos', 'kgp-talkie']
  ```

---

## Regular Expression in spaCy  

### Example: Extracting "Google I/O"  
```python
text = 'Google announced a new Pixel at Google I/O Google I/O is a great place to get all updates from Google.'
```

**Code:**  
```python
pattern = [{"TEXT": {"REGEX": "Google I/O"}}]
callback_method = ...  # Define your callback
```

**Matches:**  
```python
[(11578853341595296054, 6, 10), (11578853341595296054, 10, 14)]
```

---

**Source:** https://kgptalkie.com/rule-based-phrase-text-extraction-and-matching-using-spacy