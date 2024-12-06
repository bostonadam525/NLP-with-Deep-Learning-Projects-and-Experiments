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


# How to overcome issuses with Seq2Seq?
1. You can pass a concatenation of all the hidden states from the encoder to the decoder block (e.g. mean pooling)
  * Problem with this is LARGE MEMORY CONSUMPTION
2. Focus attention in the right place
  * This means that we combine all hidden states into a transition state called the “attention layer”
  * Decoder can then refer to the “attention layer” which is all hidden states concatenated and stored from the previous encoder block there to output a more informative decoder
      * This is called “cross attention” (encoder based attention)
  * Decoder can also just focus on the previous hidden state which is “self attention"

Example (souce: udemy):
![image](https://github.com/user-attachments/assets/a61b0c49-d5c3-4007-b137-86e07f4fbe86)













  


