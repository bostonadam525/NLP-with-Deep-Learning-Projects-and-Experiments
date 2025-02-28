## 1. get training history
trainer_history_all = trainer.state.log_history
trainer_history_metrics = trainer_history_all[:-1]
trainer_history_training_time = trainer_history_all[:-1]

# View first 3 outputs
trainer_history_metrics[:3]


## 2. extract training metrics
import pprint

# Extract eval and training metrics
trainer_history_training_set = []
trainer_history_eval_set = []

# Loop through our metrics
for item in trainer_history_metrics:
  item_keys = list(item.keys())
  if any("eval" in item for item in item_keys):
    trainer_history_eval_set.append(item)
  else:
    trainer_history_training_set.append(item)

# Show first item from each
print(f"First item in training set:")
pprint.pprint(trainer_history_training_set[:2])

print(f"\nFirst two items in eval epochs:")
pprint.pprint(trainer_history_eval_set[:2])


## 3. create pandas df for training and eval metrics
trainer_history_train_df = pd.DataFrame(trainer_history_training_set)
trainer_history_eval_df = pd.DataFrame(trainer_history_eval_set)

## trainer df head - train data
trainer_history_eval_df.head()



## 4. plot loss curves
import matplotlib.pyplot as plt
%matplotlib inline

plt.figure(figsize=(12,6))
plt.plot(trainer_history_train_df['epoch'], trainer_history_train_df['loss'], label="Training loss")
plt.plot(trainer_history_eval_df['epoch'], trainer_history_eval_df['eval_loss'], label="Evaluation loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Multiclass Text Classification fine-tuning ModernBERT training and evaluation loss over time")
plt.legend()
plt.show();
