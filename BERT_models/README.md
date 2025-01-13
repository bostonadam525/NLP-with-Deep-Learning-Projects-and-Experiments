# BERT Models
* BERT - (Bidirectional Encoder Representations from Transformers)


## BERT paper
* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
* introduced 2 things:

1. MLM - Masked Language Modeling
2. NSP - Next Sentence Prediction


## Overview
* layers of encoders of Transformer model. 

* BERTbase
  * 12 encoder layers
  * size of hidden size of feedforward layer is 3072.
  * 12 attention heads. 

* BERTLarge
  * 24 encoder layers
  * size of hidden size of feedfoward layer is 4096.
  * 16 attention heads


## Differences with vanilla transformer
* Embedding vectors (embeddings) is 768 and 1024 for 2 models
* Positional embeddings are absolute and learnt during training and limited to 512 positions. 
* linear layer head changes according to application.
* BERT uses WordPiece Tokenizer for sub-word tokens. Vocab size is ~30,000 tokens. 


## Inputs to BERT
1. Input
  * WordPiece/Sub-word tokenizer applied
  * `[CLS]` or `[SEP]` tokens applied

2. Token embeddings
  * Token encoding vectors (e.g. Word2Vec)

3. Segment Embeddings
  * Sentence segementation 

4. Positional Embeddings
  * learned sequential positions —numeric ids

5. Final embeddings
  * sum of all previous layer embeddings

![image](https://github.com/user-attachments/assets/de4bde46-3bda-4bc5-a476-9ad85da2a878)



## Output layer depends on specific task head applied
1. Linear layer — model head
  * Classification head (neurons dependent upon number of classes or tasks)
  * NER —> token classification 

2. Softmax
3. Output probabilities
   * output layer —> 768 dim
   * Token embeddings (T) — context vectors 
   * Context vector for entire sentence (C)

![image](https://github.com/user-attachments/assets/10db71cb-a5be-4f58-953b-34c1d570dfdc)

  
* Excellent post by Jay Alammar on Hidden States - [Finding the Words to Say: Hidden State Visualizations for Language Models](https://jalammar.github.io/hidden-states/)
* Excellent post by Jay Alammar on BERT for Classification - [A Visual Guide to Using BERT for the First Time](https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)
* Example of a specific classification head on BERT (logistic regression) as demonstrated from Jay Alammar's post:

![image](https://github.com/user-attachments/assets/a7fdde75-d51c-4e9c-a2d0-69e61d5085a7)

  
## BERT - Generally speaking, 2 loss functions are used
1. MLM — Masked Language Modeling
  * Cross Entropy Loss (multi-class prediction)
2. NSP - Next Sentence Prediction
  * Binary loss (binary prediction)


## Masked Language Modeling (MLM)
* Pre-training procedure selects 15% of tokens from sentence to be masked.
* When a token is selected to be masked from a sentence:
      * 80% of time it is replaced by `[MASK]` token
      * 10% of time it is replaced by a random token
      * 10% of time is is NOT replaced. 

* As example sentence: “People go skiing in Colorado in the winter.”
      * The token “skiing” would be 80% of the time replaced by `[MASK]`
      * 10% of the time replaced by a random token such as: “fishing”
      * 10% of time not replaced. 




## Context in BERT
* Bi-directional models excel at CONTEXT.
* BERT models allow for contextual understanding before and after entities in a phrase or sentence.
      * This is what a bi-directional model excels at. 
      * They give a global view of the sentence and the semantic contextual importance of words.
* Decoder models like GPT excel at next token or word prediction  (e.g. look left predict right). 
* Encoder vs. Decoder architecture - [source](https://medium.com/analytics-vidhya/paper-summary-bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding-861456fed1f9)

![image](https://github.com/user-attachments/assets/43d44c4d-d29d-4794-b5fa-a96c4cf94d65)


### Semantic meaning is important, As an example
* Sentence 1: …..they were on the right side of the street
* Sentence 2: …..they were on the right side of history. 
    * “right” has different meanings based on the context. 
    * “on the” and “side of” are the “fixed windows” with the attention term “right”.

* Thus, direction is important!!!
* RNN models were only looking in 1 direction (forward RNN)

