# Torchmetrics scripts
%%capture
!pip install torchmetrics

## 1. Accuracy
from torchmetrics import Accuracy
import torch

## set num of classes 
num_classes=4

# Setup metric
# Assuming 'cuda' is your desired device, change if necessary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
y_blob_test = y_blob_test.to(device) # Move y_blob_test to the device
torchmetric_accuracy = Accuracy(task="multiclass",
                                num_classes=num_classes).to(device)

# Calculate accuracy
torchmetric_accuracy(y_preds, y_blob_test)

-------------------------------------------------------------------------------

## 2. Precision, Recall, F1 Score
import torch
from torchmetrics import Precision, Recall, F1Score

## num classes
num_classes=4


# Setup metric
# Assuming 'cuda' is your desired device, change if necessary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
y_blob_test = y_blob_test.to(device) # Move y_blob_test to the device


## Precision
torchmetric_precision = Precision(task="multiclass",
                                  num_classes=num_classes).to(device)

## Recall
torchmetric_recall = Recall(task="multiclass",
                            num_classes=num_classes).to(device)

## F1 score
torchmetric_f1score = F1Score(task="multiclass",
                              num_classes=num_classes).to(device)

## calculate metrics
precision_score = torchmetric_precision(y_preds, y_blob_test)
recall_score = torchmetric_recall(y_preds, y_blob_test)
f1_score = torchmetric_f1score(y_preds, y_blob_test)

## print each
print(f"Precision score: {precision_score:.3f}%")
print(f"Recall score: {recall_score:.3f}%")
print(f"F1 Score: {f1_score:.3f}%")

----------------------------------------------------------------

## 3. AUROC
from torch import tensor
from torchmetrics.classification import MulticlassAUROC


## Setup AUROC metric
torchmetric_AUROC = MulticlassAUROC(num_classes=4, average="macro", thresholds=None).to(device)

## Calculate AUROC
torchmetric_AUROC(y_pred_probs, y_blob_test)
