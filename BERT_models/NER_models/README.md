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

