# DistilBERT -- Knowledge Distillation for BERT

* These 3 models are all versons of “distilled BERT"
1. DistilBERT
2. MobilBERT
3. TinyBERT 


## DistilBERT
* DistilBERT, simply put is a distilled version of BERT: smaller, faster, cheaper and lighter
* Original 2019 paper by San et al.: https://arxiv.org/abs/1910.01108

### Number of encoders for BERT models
* BERT large —> 24
* BERT base —> 12
* DistilBERT —> 6
  * Uses teacher-student learning process

### DistilBERT's key methodologies are the following:

1. Knowledge Distillation

2. Triple Loss Function

3. Architecture Modification


### Knowledge distillation is very common in Machine Learning 
* The basic concept: a smaller (student) model is trained to mimic and learn from the performance of a larger (teacher) more complex model. 
* This is often done for:
  * Model compression
  * Inference Speedup
  * Deployment efficiency 

* Multiple loss functions are used in this process:
  1. distillation loss
  2. student loss
  3. cosine sim loss

* How distillation works
  1. Weights passed to teacher and student
  2. Teacher weights frozen
  3. Student model trained



### Why do we need DistilBERT?
* 60% faster than original BERT
* 44 million fewer parameters —> 40% smaller
* Maintains 97% performance of original BERT
* Able to scale this model in production alot easier and more efficiently than standard BERT model. 
* **Able to easily use DistlBERT on CPU —> major advantage!!**


### DistilBERT Architecture
1. **Encoder Layers reduced from 12 in BERT base to 6 in DistilBERT!!!!**
2. Student model (DistilBERT) trained to predict probability distribution of vocabulary from teacher model (BERT) using the same input text
3. Student learns to replicate attention patterns of teacher
4. In training, optimization includes --> temperature scaling applied to Softmax outputs
5. Token type embeddings removed, Pooler removed --> Can’t perform next sentence predictions
6. Same corpus used in training: English Wikipedia and Toronto Book Corpus



### DistilBERT Loss Functions
* During training, DistilBERT learns from BERT and updates its weights via these 3 loss functions:

1. **Masked language modeling (MLM) loss**
 * Higher SoftMax Temperature value == softer distribution 
 * As T value increases —> variation between token probabilties is smaller (e.g. T=1) —> Probabilistic
 * Higher variation in probability distribution when temperature is less (e.g. T=0.1) —> Deterministic
 * If you recall, **Temperature** is a hyperparameter of LSTMs (and neural networks in general) used to control the randomness of predictions by scaling the logits before applying a softmax function.
   * Temperature scaling has been widely used to improve performance for NLP tasks that utilize the Softmax decision layer.
   * Temperature scaling increases the randomness of the probability distribution.
   * This characterizes the entropy of the probability distribution used for sampling, in other words, it controls how surprising or predictable the next word will be.
   * The scaling is done by dividing the logit vector by a value T, which denotes the temperature, followed by the application of softmax. source: [Softmax Temperature](https://medium.com/@harshit158/softmax-temperature-5492e4007f71)




![image](https://github.com/user-attachments/assets/602382d4-a8e1-4bf7-9b2e-6bfbcba0dd54)




2. **Distillation Loss**
 * Prob distribution of Teacher vs. Student compared via cross-entropy


3. **Similarity Loss**
 * Embeddings of Teacher and Student computed together and similarity compared —> weights updated


* Final loss calculation
  * `final loss = MLM loss + distillation loss + similarity loss`



## MobileBERT
* Compact Task-Agnotic BERT for Resource Limited Devices
* Original 2020 arixiv paper: https://arxiv.org/abs/2004.02984



### What exactly is mobileBERT
* Compresses and accelerates BERT
   * Allows deployment on mobile devices with limited resources
   * Maintains same high performance
* Versatile and Task-agnostic model
   * You can fine-tune this model for pretty much any NLP task without task-specific modifications. 
* “Thin” version of BERTLarge to reduce computation load:
   * Bottleneck structures
   * Balances self-attention
   * Feed-forward networks 
* Transfer learning was used to train model
   * knowledge transfer from BERTLarge (IB-BERT) model
   * Ensured smaller model retains high peformance 
* Model is 4.3 smaller and 5.5 times faster than BERTBASE
   * Benchmarks performed well on:
      * GLUE
      * SQuAD


### Parameter Settings of MobileBERT
* Encoder comparison

1. BERTLARGE —> 1024
2. BERTBASE —> 768
3. MobileBERT 
   * Embedding factorization was used
      * Embedding dimension reduced to 128
      * Then 1D convolution compression applied with kernel size of 3 on raw token embeddings
   * 512 dimensional output
   * Overall Parameters
      * BERTLARGE —> 334M
      * BERTBASE —> 109M
      * MobileBERT —> 25.3M
      * MobileBERTtiny —> 15.1M




## TinyBERT 
* Distilling BERT for Natural Language Understanding (NLU)
* Original 2019 paper by Jiao et al: https://arxiv.org/abs/1909.10351



### What is TinyBERT?
* Created to reduce size and improve speed of BERT while maintaining high performance on NLP tasks. 
* Uses a unique Transformer distillation method to effectively transfer knowledge from a larger BERT model to smaller TinyBERT model
* Employs a 2 stage learning process involving general distillation from a non-fine-tuned BERT and task-specific distillation from a fine-tuned BERT.
   * This enhances both general and task-specific capabilities. 
* Model has only 4 layers and achieves the following:
   * 96.8% of BERTBASE performance on GLUE benchmark 
   * 7.5 times smaller than BERTBASE
   * 9.5 times faster for inference than BERTBASE



### Compare and Contrast 3 distilled BERT models
1. All 3 models (TinyBERT, DistilBERT, MobileBERT) goal was to reduce size of original BERT model.
   * Allowed for enhanced efficiency
   * Allowed for deployment on mobile devices and devices without need for GPU (CPU and low memory support)
2. Each model uses “knowledge distillation techniques” to transfer knowledge from “Teacher” to “Student”
   * Transfer learning — knowledge transfer from LARGE model to SMALL model
   * Preserves teachers abilities (large model)
   * Reduces computational demands
3. All 3 models maintain the ability to be fine-tuned for various downstream NLP tasks
   * Very versatile across different apps without task-specific pre-training. 
4. All 3 models despite being SMALLER maintain similar performance to original BERT on benchmarks. 


