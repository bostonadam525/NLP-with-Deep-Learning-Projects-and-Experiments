## 1. hugging face model evaluation using `evaluate` library

import evaluate
import numpy as np

accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def compute_metrics(predictions_and_labels):
    predictions, labels = predictions_and_labels

    if len(predictions.shape) >= 2:
        predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels, average='weighted')
    recall = recall_metric.compute(predictions=predictions, references=labels, average='weighted')
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='weighted')

    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"]
    }

--------------------------------------------------------------------------------------------
## 2. sklearn evaluation function 

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_sklearn(predictions_and_labels):
    predictions, labels = predictions_and_labels

    if len(predictions.shape) >= 2:
        predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


-----------------------------------------------------------------------------------------------
# 3. PyTorch Lightning --- full model BERT model training with metrics
## This example includes:
## 1. A custom TextClassificationDataset for handling text data and labels.
## 2. The BERTClassifier model with training, validation, and test steps.
## 3. Data preparation, including splitting into train and validation sets.
## 4. Initialization of the tokenizer, datasets, and data loaders.
## 5. Setting up the PyTorch Lightning Trainer with logging and checkpointing.
## 6. Training and testing the model.
## 7. Printing the final metrics.

### To use this code:
## 1. Replace the placeholder texts and labels with your actual data.
## 2. Adjust the num_classes in the BERTClassifier initialization to match your task.
## 3. Modify hyperparameters like max_length, batch_size, and max_epochs as needed.
## 4. Run the script to train and evaluate your BERT model or other transformer model.
### This setup will automatically calculate and log accuracy, precision, recall, and F1 score during training and validation. 
### You can view the logs using TensorBoard for detailed tracking of these metrics over time.

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torchmetrics import Accuracy, Precision, Recall, F1Score
from sklearn.model_selection import train_test_split

# Define dataset
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# BERT Classifier
class BERTClassifier(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=2e-5):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
        self.learning_rate = learning_rate

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.precision = Precision(task="multiclass", num_classes=num_classes, average='weighted')
        self.recall = Recall(task="multiclass", num_classes=num_classes, average='weighted')
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average='weighted')

    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = self(input_ids, attention_mask)
        loss = torch.nn.functional.cross_entropy(outputs, labels)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = self(input_ids, attention_mask)
        loss = torch.nn.functional.cross_entropy(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        self.accuracy(preds, labels)
        self.precision(preds, labels)
        self.recall(preds, labels)
        self.f1(preds, labels)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.accuracy, prog_bar=True)
        self.log('val_precision', self.precision, prog_bar=True)
        self.log('val_recall', self.recall, prog_bar=True)
        self.log('val_f1', self.f1, prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

# Main execution
if __name__ == '__main__':
    # Prepare data --- example using dummy data
    texts = ["This is a positive review.", "This is a negative review.", ...]  # Your text data
    labels = [1, 0, ...]  # Your labels

    # Split the data
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create datasets
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length=128)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length=128)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Initialize model
    model = BERTClassifier(num_classes=2)  # Adjust num_classes based on your task

    # Set up logger and checkpointing
    logger = TensorBoardLogger("tb_logs", name="bert_classifier")
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max")

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=5,
        gpus=1 if torch.cuda.is_available() else None,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Test the model
    trainer.test(model, val_loader)  # You can use a separate test set here if available

    # Print final metrics
    print(f"Final Accuracy: {model.accuracy.compute():.4f}")
    print(f"Final Precision: {model.precision.compute():.4f}")
    print(f"Final Recall: {model.recall.compute():.4f}")
    print(f"Final F1 Score: {model.f1.compute():.4f}")
