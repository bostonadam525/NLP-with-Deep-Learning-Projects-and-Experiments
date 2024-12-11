# Named Entity Recognition with Transformers
* A repo dedicated to NER modeling with transformers.


# Review of NER Concepts
* As you may or may not know, NER is a powerful technique that is often used to identify and classify "named entities" such as but not limited to:
1. people
2. places
3. organizations
4. dates
5. locations
6. ...etc...

## Spacy `displaCy Entity Visualization`
* This is a good demo that spacy put together to test out NER modeling: https://demos.explosion.ai/displacy-ent


# Tokens and Tags in NER Models
* NER modeling is also most commonly known as `**Token Classification**.
* NER modeling breaks data down into tokens and tags for a model to process and classify the information.
The acronym for the schema often used is known as BILUO which stands for:
  * Begin
  * In
  * Last
  * Unit
  * Out

* The "BILUO" Schema is the most commonly used as demonstrated by the spacy library:
![image](https://github.com/user-attachments/assets/e9a486ed-793f-4ed1-a904-716c5093927d)

* However, a famous paper by Ratinov and Roth entitled "Design Challenges and Misconceptions in Named Entity Recognition"
showedthat BILOU encoding scheme significantly outperforms BIO or IOB encoding as both are more difficult for a model to learn from data. 
  * The main reason is the BILOU schema has boundary tokens making it easier for a NER model to train on.
  * Link to paper: https://aclanthology.org/W09-1119/
  * Another good NER resource: https://towardsdatascience.com/extend-named-entity-recogniser-ner-to-label-new-entities-with-spacy-339ee5979044
 
* Example of the IOB or BIO schema:
![image](https://github.com/user-attachments/assets/932658a5-3adb-4821-a4cb-0ea32100649a)



## Other NER Schemas
* There are 7 "standard" NER schemas:
There are 7 known tagging schemes:
1. IO
2. IBO/BIO
3. IOE
4. IOBES/BIOES
5. BI
6. IE
7. BIES

* This is an excellent breakdown application of these 7 schemas, source: Borelli et al. 2024.
![image](https://github.com/user-attachments/assets/a6f7967a-fca5-4740-bfa2-74163c92cf44)
* Full paper: https://arxiv.org/html/2401.10825v1/#S7.T3

## Modeling Approaches to NER
* This is an excellent review chart of the various NER modeling approaches which again comes from the above paper by Borelli et al. 2024
![image](https://github.com/user-attachments/assets/99c955cf-fd2a-4643-8232-5966b355ea8b)




# General NER Modeling Workflow
1. Text input
  * NER begins with text which can be word, sentence, paragraph, document or larger corpora. 

2. Tokenization
  * Input text is split into individual words or tokens (tokenization).
  * Tokenization is MOST important in NER models because it works on a token by token level. 

3. Entity Recognition
  * Tokenized text is analyzed to identify "spans" of tokens that correspond to named entities.
  * NER systems use various techniques such as:
    * rule-based approaches
    * machine learning models
      * conditional random fields
      * deep learning
    * combination of methods

4. Entity/Token Classification
  * After entities are recognized they are then classified into predefined categories of choice such as:
    * person
    * organization
    * location
    * date
    * number
    * product
    * company
    * ...etc...
  * The purpose of these categories is to help organize and provide context to the recognized entities. 

5. Output
   * Final output of NER model is structured representation of the original text with identified named entities and their respective categories. 
   * The output can then be used for various purposes such as but not limited to:
      * information extraction
      * content summarization
      * sentiment analysis
      * etc...


# Architecture of NER Transformer Models
* This section is sourced from the excellent paper by Marcinczuk in June 2024 entitled: **Transformer-based Named Entity Recognition with Combined Data Representation**
  * arxiv link: https://arxiv.org/html/2406.17474v1
* The original architecture described by Devlin et al. in their 2019 paper used this method:
  * NER modeled as a sequence classification task. 
  * The neural network architecture consists of two main elements: 
    * 1) Pre-trained language model (PLM) embeddings
    * 2) Classifier layer
* The PLM part generates a context-based representation of each word (a vector of the first words subtoken). 
* The classifier layer outputs one of the IOB2 labels for each word. 
  * The label set depends on a given datasets categories of named entities. 
  * In post-processing, the sequences of IOB2 labels are wrapped into continuous annotations. 
  * The text is divided into tokens, and then each token is divided into subtokens by the model tokenizer. 
  * Some tokes are divided into more than one subtoken. 
  * For example below, the Eiffel word is tokenized into three subtokens: 
    * `E`
    * `iff`
    * `el`
      * In this case, **only the first subtoken** is passed through the classifier to obtain the label of the word. 
      * This approach is adequate due to the **attention mechanism** of transform-based models.
* Representation of this architecture from the paper noted above:
![image](https://github.com/user-attachments/assets/cf452377-d0b9-494b-90fe-396e7fc49f08)


