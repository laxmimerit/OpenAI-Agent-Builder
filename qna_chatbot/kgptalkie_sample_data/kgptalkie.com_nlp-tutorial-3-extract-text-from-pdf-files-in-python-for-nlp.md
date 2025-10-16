https://kgptalkie.com/nlp-tutorial-3-extract-text-from-pdf-files-in-python-for-nlp

# Extract Text from PDF Files in Python for NLP

**Source:** https://kgptalkie.com/nlp-tutorial-3-extract-text-from-pdf-files-in-python-for-nlp

## Published by

georgiannacambel  
30 August 2020

---

## Extraction of text from PDF using PyPDF2

This notebook demonstrates the extraction of text from PDF files using **python** packages. Extracting text from PDFs is an easy but useful task as it is needed to do further analysis of the text.

---

## Working with .PDF Files

We are going to use **PyPDF2** for extracting text. You can download it by running the command given below:

```bash
!pip install PyPDF2
```

Now we will import **PyPDF2**:

```python
import PyPDF2 as pdf
```

We have used the file `NLP.pdf` in this notebook. The `open()` function opens a file, and returns it as a file object. `rb` opens the file for reading in binary mode.

```python
file = open('NLP.pdf', 'rb')
file
```

Output:
```
<_io.BufferedReader name='NLP.pdf'>
```

`PdfFileReader(file)` initializes a `PdfFileReader` object for the file handler `file`.

```python
pdf_reader = pdf.PdfFileReader(file)
pdf_reader
```

Output:
```
<PyPDF2.pdf.PdfFileReader at 0x246349895c0>
```

You can check all the operations which we can perform on `pdf_reader` using `help(pdf_reader)`.

```python
help(pdf_reader)
```

### `getIsEncrypted()`

The property shows whether the PDF file is encrypted. It is returning `False` that means the file which we are using is not encrypted.

```python
pdf_reader.getIsEncrypted()
```

Output:
```
False
```

### `getNumPages()`

Calculates and returns the number of pages in the PDF file.

```python
pdf_reader.getNumPages()
```

Output:
```
19
```

### `getPage()`

Retrieves a page by number from the PDF file.

```python
page1 = pdf_reader.getPage(0)
page1
```

`page1` has function `extractText()` to extract text from the PDF page.

```python
page1.extractText()[:1050]
```

Output:
```
'Lkit: A Toolkit for Natuaral Language Interface Construction 2. Natural Language Processing (NLP) This section provides a brief history of NLP, introduces some of the main problems involved in extracting meaning from human languages and examines the kind of activities performed by NLP systems.   2.1. Background Natural language processing systems take strings of words (sentences) as their input and produce structured representations capturing the meaning of those strings as their output. The nature of this output depends heavily on the task at hand. A natural language understanding system serving as an interface to a database might accept questions in English which relate to the kind of data held by the database. In this case the meaning of the input (the output of the system) might be expressed  in terms of structured SQL queries which can be directly submitted to the database.  The first use of computers to manipulate natural languages was in the 1950s with attempts to automate translation between Russian and English [Locke & Booth]'
```

```python
page2 = pdf_reader.getPage(1)
page2.extractText()
```

Output:
```
'Lkit: A Toolkit for Natural Language Interface Construction     2.2. Problems Two problems in particular make the processing of natural languages difficult and cause different techniques to be used than those associated with the construction of compilers etc for processing artificial languages. These problems are (i) the level of ambiguity that exists in natural languages and (ii) the complexity of semantic information contained in even simple sentences.   Typically language processors deal with large numbers of words, many of which have alternative uses, and large grammars which allow different phrase types to be formed from the same string of words. Language processors are made more complex because of the irregularity of language and the different kinds of ambiguity which can occur. The groups of sentences below are used as examples to illustrate different issues faced by language processors. Each group is briefly discussed in the following section (in keeping with convention, ill-formed sentences are marked with an asterix).  1.  The old man the boats.  2.  Cats play with string.  * Cat play with string.  3.  I saw the racing pigeons flying to Paris.   I saw the Eiffel Tower flying to Paris.  4.  The boy kicked the ball under the tree.   The boy kicked the wall under the tree.   1. In the sentence "The old man the boats" problems, such as they are, exist because the word "old" can be legitimately used as a noun (meaning a collection of old people) as well as an adjective, and the word "man" can be used as a verb (meaning take charge of) as well as a noun. This causes ambiguity which must be resolved during syntax analysis. This is done by considering all possible syntactic arrangements for phrases and sub-phrases when necessary.   The implication here is that any parsing mechanism must be able to explore various syntactic arrangements for phrases and be able to backtrack and rearrange them whenever necessary.   2 '
```

---

## Append, Write or Merge PDFs

`PdfFileWriter()` class supports writing PDF files out, given pages produced by another class typically `PdfFileReader()`.

```python
pdf_writer = pdf.PdfFileWriter()
```

`addPage()` adds a page to the PDF file. The page is usually acquired from a `PdfFileReader` instance.

```python
pdf_writer.addPage(page2)
pdf_writer.addPage(page1)
```

Now we are going to open a new file `Pages.pdf` and write the contents of `pdf_writer` to it. `pdf_writer.write()` writes the collection of pages added to `pdf_writer` object out as a PDF file. `close()` closes the opened file.

```python
output = open('Pages.pdf','wb')
pdf_writer.write(output)
output.close()
```

`Pages.pdf` has the following pages:

- Page 1: Content from `page2`
- Page 2: Content from `page1`

**Source:** https://kgptalkie.com/nlp-tutorial-3-extract-text-from-pdf-files-in-python-for-nlp

**Source:** https://kgptalkie.com/nlp-tutorial-3-extract-text-from-pdf-files-in-python-for-nlp