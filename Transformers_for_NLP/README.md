# Transformers for NLP
* Repo all about Transformers for NLP - experiments, use cases, demos, tutorials.
* A well known open-source resource for transformers is Huggingface.


## Huggingface Workflow
* This is a typical huggingface workflow (source: [Daniel Bourke](https://www.learnhuggingface.com/)


![image](https://github.com/user-attachments/assets/28373d00-177a-4a57-83e1-cbdf4a723a85)






# Seq2Seq Models
* Introduced by Google in 2014
* Maps variable length sequences into fixed-length memory (which is their problem!)
* Inputs and outputs have different lengths
  * You can input 3 words and output 5 words, etc..
* LSTMs and GRUs are used in Seq2Seq models to avoid vanishing and exploding gradient problems

* Main takeaway:
  * information is processed in sequences, (think of time steps in a time series)
  * Words and sentences have sequences just like time steps.
  * Words are input to encoder --> processed sequentially --> sent to hiddent state --> Sent to decoder --> output tokens.
    * Example (source: udemy)
    * Note about image below. "Hidden State 4" or "H4" is the final hidden state from the encoder that gets passed to the decoder.
   
  ![image](https://github.com/user-attachments/assets/2f61c5ed-78b1-4372-a3cd-d827653e7d15)

## Encoder Block of Seq2Seq Model
* A big problem with Seq2Seq models is information loss.
  * This is because there are multiple independent LSTM layers with independent hidden states.
  * So for example, if you have 4 words or 4 tokens that is 4 LSTM layers with 4 hidden states (h1, h2, h3, h4).
  * Information is passed from h1 --> h2 --> h3 --> h4 and along the way information is lost. Why? We output 1 final single hidden state (h4) which is compressed information of the previous 3 hidden states, hence the information is lost.
  * Imagine if you had more than 4 words or 4 tokens and instead had paragraphs of information and compressed it into 1 hiddent final state, think about how much information is lost then! See image below (source: udemy)
* Below we can see the **Encoder** block of the Seq2Seq model:

![image](https://github.com/user-attachments/assets/f2cc4a14-c22d-4c81-8d0c-680fe168862d)


## Decoder block of Seq2Seq Model
Decoder
* H4 final output state from encoder sent to decoder.
* Start of sequence token <SOS> kicks of the decoder blocks which are the same as encoder with multiple LSTM blocks. 
* Output of previous state is input to next state. 
* <EOS> end of sequence is the last token generated.
* Example below (source: udemy):

![image](https://github.com/user-attachments/assets/9d5f2788-8bcc-4f20-aef2-64543630e37d)

## Issues with Seq2Seq
1. Variable length sentences + fixed-length memory (single hidden states with compressed information)
2. As sequence size increases —> model performance decreases!
3. Information Bottleneck occurs as information is compressed into final decoder output
4. **"Vanishing Gradients" Result**
   * This is a situation where the gradients used to update network weights during backpropagation become extremely small as they travel through the layers, particularly in the earlier layers of a network, hindering the learning process and making it difficult for the model to **capture long-range dependencies** between input and output sequences


## How to overcome issuses with Seq2Seq?
1. You can pass a concatenation of all the hidden states from the encoder to the decoder block (e.g. mean pooling)
  * Problem with this is LARGE MEMORY CONSUMPTION
2. Focus attention in the right place
  * This means that we combine all hidden states into a transition state called the “attention layer”
  * Decoder can then refer to the “attention layer” which is all hidden states concatenated and stored from the previous encoder block there to output a more informative decoder
      * This is called “cross attention” (encoder based attention)
  * Decoder can also just focus on the previous hidden state which is “self attention"

Example (souce: udemy):
![image](https://github.com/user-attachments/assets/a61b0c49-d5c3-4007-b137-86e07f4fbe86)

* Original paper on Attention was published in 2015 (not the transformer one that was in 2018)
  * paper: Neural Machine Translation By Jointly Learning to Align and Translate
  * Link: https://arxiv.org/abs/1409.0473
* Takeaways
  * As sentence length increases —> Traditional Seq2Seq model loses information significantly
  * As sentence length increases —> Seq2Seq with Attention maintains the same performance and virtually no information loss. 
    * Can focus on specific hidden states during this!

* Performance metrics:
1) BLEU Score — Bi-lingual evaluation understudy (usually translation metric)
Precision Focused — the “correctness” of the positive predictions made by the model.
2) ROUGE Score — Recall oriented understudy for gisting evaluation (usually summarization metric)
Recall Focused — the “ACTUAL” positive cases the model correctly identifies.
3) Combine BLEU + ROUGE —> F1 Score = 2 * ((P*R) / (P+R))
F1 score is the harmonic mean of Precision and Recall

## How to use all the hidden states?
1. Add all hidden states together (pointwise addition with normalization)
  * The problem with this is there is no weight is applied.
    * Equation: c = h1 + h2 + h3 + h4
  * To solve this you pass the previous decoder hidden state. 
    * Equation: c = w1h1 + w2h2 + w3h3 + w4h4



# Queries, Keys, Values
* We should first remember the 2018 paper "Attention is All You Need" the original Transformers paper, which you can read here:  https://arxiv.org/abs/1706.03762
* Value, Key and Query Vectors are important building blocks of transformer models as we see below:

![image](https://github.com/user-attachments/assets/746cfea8-9d44-4c8e-8810-23daca36ffed)

* Source: "Attention is All You Need", 2018

## How does this work? 
* Let’s say you have a sentence or phrase:
  * “It’s time for coffee”
* Now let’s say you are searching for a YouTube video, there is:
  * Query —> Search term
  * Key —> Videos Title
  * Value —> Videos Content
* Attention is calculated using the lookup table. 
  * Cosine Similarity is used to match the key, query, and value
* Equation for Attention is:
  * Attention (Q, K, V) = Softmax ((Q*KT)/ np.sqrt(dk)) * V
    * d = dimension of key vector
* Queries, Keys, Values with weights
  * Equation: q1w1 + q1w2 + q1w3 + q1w4
  * Note: These are attention weights



# Scaled Dot Product Attention
* This is the multi-head attention layer that consists of several attention layers running in parallel
These are the multiple values and layers of Q,K,V

* Standard Transformer architecture is the following: 

1. Queries, Keys, Values —> Vector space
2. Matrix Multiplication 
3. Query and Key MatMul
4. Scaling
5. Masking (optional)
6. SoftMax 
7. Output layer with MatMul for final --> Q, K, V

* Depiction of above from the original paper:

![image](https://github.com/user-attachments/assets/caf48838-a68c-44af-b054-7ef6633fdc5a)

* This is an excellent review of the scaled dot product attention mechanism: https://medium.com/@vmirly/tutorial-on-scaled-dot-product-attention-with-pytorch-implementation-from-scratch-66ed898bf817

* Alignment Weights
   * Concept is that similar words have LARGE weights


* Flexible Attention
   * This works for languages with different grammar structures. 
   * Attention focuses where it needs to. 
   * Does NOT depend on the position (as in n-gram models).



# Transformers with encoding and decoding stacks

## Encoder stack - Flow
1. tokenized text
2. token encodings (numerical vectors)
3. token embedding vectors + positional embeds
  * **Generic vectors**
  * Positional Encoding
     * Very little variation
     * Sinusoidal function
     * Ordered position information
4. encoder stack
  * multiple encoders stacked on each other
  * Multi-head attention network (many single attention networks in parallel)
5. final hidden states
  * **Contextual vectors**
6. K,V

### Encoder Layer
* 3 main inputs

1. Query
2. Key
3. Value

* Inputs to multi-head attention layer
   * Self Attention
* Multiple feed forward networks
* Output 


![image](https://github.com/user-attachments/assets/ed8c3940-282b-437b-85e7-b3e425e14925)



## Similarity between encoder and decoder stacks
* tokenized text —> token encodings (numeric vectors) —> token embeds + positional embeds 
* Difference is that one is fed to encoder stack, the other fed to decoder stack


## Decoder stack
* Receives K,V pairs from encoder
   * —> hidden states
   * —> classification head
      * multi-class classification
      * probability of each token belonging to each class
      * goal is to predict next token or tokens
   * —> token predictions
      * result of classification head decoding raw logit probabilities 
   * —> output tokens
   * passed back to tokenzed text —> token encodings —> token embeds + positional embeds 
      * Text is presented as numeric vector encodings 
   * —> decoder stack…..back to hidden states…repeat 

![image](https://github.com/user-attachments/assets/e2676410-1407-463e-a190-b5461b2df7aa)


* Full Encoder-Decoder Architecture - [source](https://vitalflux.com/transformer-architecture-types-explained-with-examples/)

![image](https://github.com/user-attachments/assets/3fd1ce00-c6d8-4465-9fbe-446a1acfc5fc)



# What are the 3 Attention types in Transformers?
1. **Self-Attention**
  * attention between inputs only —> “self-attention”
  * Queries, keys, values come from the same input sentence.
  * Meaning of each word within a sentence is encoded in self-attention.
  * Attention of a word to itself = 1

2. **Masked-Self Attention**
  * attention in decoder output but only in previously generated tokens —> “masked-self attention"
  * Q, K, V come from the same sentence. Queries don't attend to future positions.
  * 1) Used in decoder transformers to predict transformer outputs
       * If have tokens 1, 2, 3 we want to predict 4
       * Attention comes from tokens 1, 2, 3 (previous tokens)
* 2) Causal modeling to avoid data leaks 



3. **Cross-Attention (encoder-decoder attention)**
   * attention between input and output
   * Queries from one sentence, keys and values from another. 
     1) mostly used in translation tasks
     2) Cross attention —> translation across the matrix
   * Key differences between self-attention and cross-attention
       * Self-attention works on a single sequence by generating contextual embeddings for its tokens.
       * Cross attention handles two sequences at once.
           * For example, in machine translation, if we are translating the sentence “We are friends” from English to Hindi, cross attention takes both the input sequence (“We are friends”) and the output sequence (“हम दोस्त हैं”) to establish the relationships between these sequences.
           * It computes how strongly each word in the English sentence relates to each word in the Hindi sentence.

![image](https://github.com/user-attachments/assets/4c27f759-bdd0-4e18-bda5-e7946559e674)

* In self-attention, we only need to pass one sequence as input, but for cross attention, we pass both sequences, the source and the target, making this the first point of difference between the two mechanisms. --> [source](https://medium.com/@sachinsoni600517/cross-attention-in-transformer-f37ce7129d78)




* **Applications of cross-attention**
  * Machine Translation: Translating text from one language to another involves comparing and aligning words between the source and target languages.
  * Question Answering: Determining which parts of the context are relevant to answer a given question.
  * Image Captioning: Generating descriptive text for images, where the image and text are treated as different modalities.
  * Text-to-Image Generation: Creating images based on textual descriptions.
  * Text-to-Speech: Generating speech from text, where the input is text and the output is speech.




# Multi-head attention
* apply attention multiple times or iterations
* allows a model to learn multiple aspects of contextual and semantic relationships between words. 
* Same multi-head attention used in encoder and decoder networks
* In the original paper “Attention is all you need” multi-head attention was used 8 times, thus 8 iterations. 
   * The concept was to have multiple learned linear transformations (feed forward neural networks) of input Q, K, V with attention performed in parallel.
   * From the original paper: *In this work we employ h = 8 parallel attention layers, or heads. For each of these we use dk = dv =dmodel/h = 64. Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.* ~Attention is All You Need
* Then lastly, the final concatenation with scaled-dot product attention
* This is a great representation of this:

![image](https://github.com/user-attachments/assets/d8982efd-9e33-460e-bb5a-b0275c61cc62)



## Multi-head attention Q, K, V vector spaces
* Multiple vector space dimensions are produced from the multiple linear transformations (feed forward networks) via scaled-dot product
* [This is an excellent blog post that explains this in detail with code](https://data-science-blog.com/blog/2021/04/07/multi-head-attention-mechanism/)
