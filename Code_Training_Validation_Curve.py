import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# File path to your results CSV
log_path = r"C:\\Users\\VIT\\Desktop\\Model_Result_File\\results246.csv"

# Loading the CSV file
data = pd.read_csv(log_path)

# Extracting metrics from the CSV
epochs = data['epoch']
train_box_loss = data['train/box_loss']
train_cls_loss = data['train/cls_loss']
train_dfl_loss = data['train/dfl_loss']
val_box_loss = data['val/box_loss']
val_cls_loss = data['val/cls_loss']
val_dfl_loss = data['val/dfl_loss']
precision = data['metrics/precision(B)']
recall = data['metrics/recall(B)']
map50 = data['metrics/mAP50(B)']
map50_95 = data['metrics/mAP50-95(B)']
lr_pg0 = data['lr/pg0']

y_true = [0, 1, 0, 2, 1, 2, 1, 0]  
y_pred = [0, 0, 0, 2, 1, 2, 1, 1]  

# Generating classification report
report = classification_report(y_true, y_pred, output_dict=True)

# Printing the classification report
print("Classification Report:")
print(report)

# Plotting Training and Validation Losses
plt.figure(figsize=(15, 10))

# Training and Validation Losses
plt.subplot(2, 2, 1)
plt.plot(epochs, train_box_loss, label='Train Box Loss')
plt.plot(epochs, train_cls_loss, label='Train Cls Loss')
plt.plot(epochs, train_dfl_loss, label='Train DFL Loss')
plt.plot(epochs, val_box_loss, label='Val Box Loss')
plt.plot(epochs, val_cls_loss, label='Val Cls Loss')
plt.plot(epochs, val_dfl_loss, label='Val DFL Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()

# Metrics (Precision, Recall, mAP)
plt.subplot(2, 2, 2)
plt.plot(epochs, precision, label='Precision (B)')
plt.plot(epochs, recall, label='Recall (B)')
plt.plot(epochs, map50, label='mAP@50 (B)')
plt.plot(epochs, map50_95, label='mAP@50-95 (B)')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.title('Precision, Recall, and mAP Metrics')
plt.legend()

# Learning Rate
plt.subplot(2, 2, 3)
plt.plot(epochs, lr_pg0, label='Learning Rate PG0')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.legend()

# Adjusting layout and show plot
plt.tight_layout()
plt.show()