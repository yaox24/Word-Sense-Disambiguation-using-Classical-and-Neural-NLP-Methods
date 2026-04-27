# Word Sense Disambiguation using Classical and Neural NLP Methods

## Overview

This project explores **Word Sense Disambiguation (WSD)**, the task of determining the correct meaning of a word based on its context. The project compares classical NLP methods with neural approaches using embeddings and machine learning.

WSD is an important problem in natural language processing, with applications in machine translation, information retrieval, and question answering.

---

## Objective

* Predict the correct sense of ambiguous words in context
* Compare baseline methods with neural approaches
* Analyze strengths and limitations of different techniques

---

## Dataset

* **SemEval 2013 Shared Task #12 dataset**
* Lexical resource: **WordNet 3.0**
* Includes:

  * Development set (~194 instances)
  * Test set (~1450 instances) 

---

## Methods

### 1. Most Frequent Sense (Baseline)

* Always selects the most common WordNet sense
* Strong baseline with no context usage

### 2. NLTK Lesk Algorithm

* Overlap-based method using:

  * context words
  * WordNet glosses
* Uses preprocessing (lowercasing, stopword removal) 

### 3. GloVe + MLP (Unsupervised)

* Uses pretrained GloVe embeddings (100d)
* Trains an MLP using self-supervised signals
* Learns to score correct sense-context pairs

### 4. GloVe + MLP (Supervised)

* Uses labeled data from dev set
* Feature vector:

  * context vector
  * sense vector
  * difference vector
* Trained with binary cross-entropy

---

## Evaluation

Models are evaluated using:

* Accuracy on dev and test sets
* Correct prediction = predicted sense matches any gold label

---

## Results

| Method                     | Dev Accuracy | Test Accuracy |
| -------------------------- | ------------ | ------------- |
| Most Frequent Sense        | 0.6753       | 0.6234        |
| NLTK Lesk                  | 0.3814       | 0.4159        |
| GloVe + MLP (Unsupervised) | 0.3814       | 0.3876        |
| GloVe + MLP (Supervised)   | 0.6546       | 0.4669        |

👉 The **Most Frequent Sense baseline performed best**, confirming its strength in real-world text. 

---

## Key Insights

* Simple baselines can outperform complex models
* Neural models require more data to generalize well
* Context representation (averaging embeddings) limits performance
* Supervised models can overfit on small datasets

---

## Tech Stack

* Python
* NLTK (WordNet, Lesk)
* NumPy
* GloVe embeddings
* Custom MLP implementation

---

## Project Structure

* `wsd-3.py` → main implementation
* `WSD_report.pdf` → detailed analysis and results

---

## How to Run

1. Install dependencies

```bash
pip install nltk numpy
```

2. Download required resources

* WordNet (via NLTK)
* GloVe embeddings (100d)

3. Run the script

```bash
python wsd-3.py
```

---

## Key Takeaways

* Compared classical NLP and neural methods in a unified framework
* Implemented both unsupervised and supervised learning approaches
* Highlighted limitations of simple embedding-based representations

---

## Notes

This project was completed as part of COMP550 (Natural Language Processing) and is presented here as a portfolio project.
