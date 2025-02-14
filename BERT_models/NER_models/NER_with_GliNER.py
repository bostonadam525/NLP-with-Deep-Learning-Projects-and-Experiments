from tqdm import tqdm
import numpy as np 
from gliner import GLiNER

# Load the model from hugging face
model = GLiNER.from_pretrained("knowledgator/gliner-multitask-v1.0")

# Define the labels
labels = [<your labels go here>]

def label_text(text):
    entities = model.predict_entities(text, labels)
    # If entities are found, return the label of the first entity
    # If no entities are found, return "Other"
    return entities[0]['label'] if entities else "Other"

# Enable tqdm for pandas
tqdm.pandas()

# Apply to df and text column containing text to be labeled
df['label'] = df['text'].progress_apply(label_text)

# Print the first few rows to check the results
print(df[['text', 'label']].head())

# Get a count of each label
print(df['label'].value_counts())
