# Part-of-Speech (POS) Tagging

## Introduction
Part-of-Speech (POS) Tagging is a fundamental task in Natural Language Processing (NLP) that involves assigning a part of speech (e.g., noun, verb, adjective) to each word in a sentence. It helps machines understand the grammatical structure of text and is a crucial step in many NLP applications like text parsing, information extraction, and machine translation.

## Objective
This project demonstrates the basic concept of POS tagging using simple examples. It uses either a rule-based, statistical, or pre-trained model to tag each word in a sentence with its respective part of speech.

## What are Parts of Speech?
Here are som common parts of speech used in tagging:
| POS Tag | Description      | Example        |
| ------- | ---------------- | -------------- |
| NN      | Noun, singular   | `cat`, `car`   |
| NNS     | Noun, plural     | `cats`, `cars` |
| VB      | Verb, base form  | `run`, `eat`   |
| VBD     | Verb, past tense | `ran`, `ate`   |
| JJ      | Adjective        | `big`, `blue`  |
| RB      | Adverb           | `quickly`      |
| PRP     | Personal pronoun | `he`, `she`    |
| IN      | Preposition      | `in`, `on`     |


## How it Works
1.  **Tokenization:** Break down the sentence into individual words
2.  **Tagging:** Assign each word a POS tag using a tagging method such as
    - Rule-based taggers
    - Statistical models (like Hidden Markov Models)
    - Machine Learning Models (like Decision Trees)
    - Pre-built libraries (e.g., NLTK, spaCy)
      
## Example
``` Python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

sentence = "The quick brown fox jumps over the lazy dog"
tokens = nltk.word_tokenize(sentence)
tagged = nltk.pos_tag(tokens)

print(tagged)

```
Output:
```
[('The', 'DT'), ('quick', 'JJ'), ('brown', 'JJ'), ('fox', 'NN'),
 ('jumps', 'VBZ'), ('over', 'IN'), ('the', 'DT'), ('lazy', 'JJ'), ('dog', 'NN')]
```

## Use Cases
- Gammer checking
- Named Entity Recognition (NER)
- Question Answering Systems
- Machine Translation
- Text-to-Speech Systems
