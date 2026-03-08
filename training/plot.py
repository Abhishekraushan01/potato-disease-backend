import pickle
import matplotlib.pyplot as plt
import os

# -----------------------------
# PROJECT ROOT PATH
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HISTORY_PATH = os.path.join(BASE_DIR, "history.pkl")

# -----------------------------
# LOAD TRAINING HISTORY
# -----------------------------
if not os.path.exists(HISTORY_PATH):
    print("history.pkl not found. Run train.py first.")
    exit()

with open(HISTORY_PATH, "rb") as f:
    history = pickle.load(f)

acc = history['accuracy']
val_acc = history['val_accuracy']
loss = history['loss']
val_loss = history['val_loss']

epochs_range = range(len(acc))

# -----------------------------
# PLOT ACCURACY & LOSS
# -----------------------------
plt.figure(figsize=(10,5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training vs Validation Accuracy')

# Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training vs Validation Loss')

plt.show()