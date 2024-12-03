# Notebooks for NLP projects using Transformers
* This folder contains NLP projects using transformers.

## Projects in this folder:
1. Fine-Tuning an `albert-base-v2` model that was pre-trained for sequence classification sentiment analysis using TextAttack and the imbdb dataset.
   * Fine-tuning dataset: kaggle video games 2024 - binary classification task
   * Fine-tuning task: Sequence Classification (unfreeze classification layer, freeze all other layers)

2. T5 pre-trained model `Michau/t5-base-en-generate-headline` which was pre-trained on 500k newspaper headlines.
   * We will utilize the model mostly out of the box from huggingface to generate newspaper headlines.
   * We will tune a few of the models parameters.
   * The output will be evaluated for quality of text generation using the METEOR score. 


3. Custom Trained DistilBERT model "food not food classifier"
  * Project from Mr. Daniel Bourke's ZTM course.
  * Full end to end project fine tuning a model from hugging face on custom dataset along with deployment to HF hub and Gradio demo app.



# Pipelines
* There is a separate folder within the **Notebooks** folder that runs through various pipelines using transformer models on hugging face for various tasks.
