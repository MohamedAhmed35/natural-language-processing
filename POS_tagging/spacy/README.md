# POS Tagging with SpaCy

## Overview
This project demonstrates how to perform Part-of-Speech (POS) tagging using spaCy, a powerful and fast NLP library for Python. POS tagging is the process of assigning each word in a sentence a label that indicates its grammatical role, such as noun, verb, or adjective.

## What is spaCy
**spaCy** is an open-source, industrial-strength NLP library designed for fast and efficient processing of large volumes of text. It comes with pre-trained models for many languages and supports tasks such as tokenization, POS tagging, named entity recognition, dependency parsing, and more.

## How POS Tagging Works with spaCy
1. **Load a pre-trained language model** (`en_core_web_sm` for English).
2. **Process** a text using the model to create a `Doc` object.
3. **Iterate** through each token in the `DOC` to access its `.pos_` and `.tag_ ` attributes.

## POS Tag Example
``` python
import spacy

# Load English tokenizer, POS tagger, etc.
nlp = spacy.load("en_core_web_sm")

# Input sentence
text = "The quick brown fox jumps over the lazy dog."

# Process the sentence
doc = nlp(text)

# Print each token and its part of speech
for token in doc:
    print(f"{token.text:<10} POS: {token.pos_:<10} Tag: {token.tag_}")
```

### Simple Output
``` yaml
The        POS: DET        Tag: DT
quick      POS: ADJ        Tag: JJ
brown      POS: ADJ        Tag: JJ
fox        POS: NOUN       Tag: NN
jumps      POS: VERB       Tag: VBZ
over       POS: ADP        Tag: IN
the        POS: DET        Tag: DT
lazy       POS: ADJ        Tag: JJ
dog        POS: NOUN       Tag: NN
.          POS: PUNCT      Tag: .
```
- `pos_`: Universal part of speech (e.g., NOUN, VERB)
- `tag_`: Detailed tag based on the Penn Treeback tagset (e.g., NN, VBZ)

## Installation
To get started, install spaCy and download teh English model:
``` bash
pip install spacy
python -m spacy download en_core_web_sm
```

## Application of POS Tagging
- Text summarization
- Named Entity Recognition (NER)
- Grammer correction tools
- Chatbots and voice assistants
- Machine translation
