# [Working with Text Files in Python for NLP](https://kgptalkie.com/nlp-tutorial-2-working-with-text-files-in-python-for-nlp)

**Source:** [https://kgptalkie.com/nlp-tutorial-2-working-with-text-files-in-python-for-nlp](https://kgptalkie.com/nlp-tutorial-2-working-with-text-files-in-python-for-nlp)

## Published by
georgiannacambel  
on 8 September 2020

---

### Working with Text Files

#### Working with f-strings for formatted print
- `f-strings` provide a concise way to embed Python expressions inside string literals.
- Example:
  ```python
  name = 'KGP Talkie'
  print(f'The YouTube channel is {name}')
  ```

#### Working with `.CSV`, `.TSV` files to read and write
- Use `pandas` to handle CSV/TSV files.
- Example:
  ```python
  import pandas as pd
  data = pd.read_csv('moviereviews.tsv', sep='\t')
  ```

#### Working with `%%writefile` to create simple `.txt` files
- **Note:** Works only in Jupyter Notebook.
- Example:
  ```python
  %%writefile text1.txt
  Hello, this is the NLP lesson.
  Please Like and Subscribe to show your support
  ```

#### Working with Python’s inbuilt file read and write
- Use `open()`, `read()`, `write()`, and `close()` methods.
- Example:
  ```python
  with open('text1.txt', 'r') as file:
      text_data = file.readlines()
  ```

---

## String Formatter

### Overview
- String formatting allows for structured output.
- Use `str.format()` or `f-strings`.

### Example with `str.format()`
```python
name = 'KGP Talkie'
print('The YouTube channel is {}'.format(name))
```

### Example with `f-strings`
```python
print(f'The YouTube channel is {name}')
```

### Alignment and Width
- `{50}` and `{10}` define column widths.
- Alignment options:
  - `:<` Left-aligned
  - `:>` Right-aligned
  - `:^` Centered

#### Example:
```python
data_science_tuts = [('Python for Beginners', 19),
                     ('Feature Selectiong for Machine Learning', 11),
                     ('Machine Learning Tutorials', 11),
                     ('Deep Learning Tutorials', 19)]

for info in data_science_tuts:
    print(f'{info[0]:<50} {info[1]:>10}')
```

---

## Working with `.CSV` or `.TSV` Files

### Reading Files
```python
import pandas as pd
data = pd.read_csv('moviereviews.tsv', sep='\t')
```

### DataFrame Overview
- `data.head()` shows first 5 rows.
- `data.shape` returns `(2000, 2)` (2000 rows, 2 columns).
- `data['label'].value_counts()` shows counts of unique values.

### Writing Files
```python
pos = data[data['label'] == 'pos']
pos.to_csv('pos.tsv', sep='\t', index=False)
```

---

## Built-in Magic Command in Jupyter: `%%writefile`

### Usage
- Writes cell content to a file.
- Example:
  ```python
  %%writefile text1.txt
  Hello, this is the NLP lesson.
  Please Like and Subscribe to show your support
  ```

### Appending to a File
- Use `-a` flag:
  ```python
  %%writefile -a text1.txt
  Thanks for watching
  ```

---

## Python’s Inbuilt File Operations

### `open()` Modes
- `"r"`: Read (default)
- `"a"`: Append
- `"w"`: Write
- `"x"`: Create (fails if file exists)

### Reading a File
```python
file = open('text1.txt', 'r')
content = file.read()
file.close()
```

### Using `with` Statement
```python
with open('text1.txt', 'r') as file:
    text_data = file.readlines()
```

### Processing Lines
```python
for temp in text_data:
    print(temp.strip())
```

### Enumerate Lines
```python
for i, temp in enumerate(text_data):
    print(f"{i}  --->  {temp.strip()}")
```

### Writing to a File
```python
with open('text3.txt', 'w') as file:
    file.write('This is third file\n')
```

### Appending Data
```python
with open('text3.txt', 'a') as file:
    for temp in text_data:
        file.write(temp)
```

---

**Watch full video here:** [https://kgptalkie.com/nlp-tutorial-2-working-with-text-files-in-python-for-nlp](https://kgptalkie.com/nlp-tutorial-2-working-with-text-files-in-python-for-nlp)