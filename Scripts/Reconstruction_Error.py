import numpy as np
import pandas as pd
import tensorflow as tf
from load_labeled_data import load_labeled_dataset
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the trained autoencoder model
autoencoder = tf.keras.models.load_model("autoencoder.keras", compile=False)

# Load the labeled test dataset (used for evaluation only)
# live = 0, spoof = 1
_, test_ds = load_labeled_dataset()

all_images, all_labels = [], []

# test dataset 
for batch in test_ds:
    images, labels = batch
    all_images.append(images)
    all_labels.extend(labels)

    
    if len(all_labels) >= test_ds.samples:
        break

# Converting lists into NumPy arrays for model input
all_images = np.concatenate(all_images)
all_labels = np.array(all_labels).astype(int)

# Use the autoencoder to reconstruct all test images
reconstructed = autoencoder.predict(all_images, batch_size=32)

# Calculate reconstruction error (Mean Squared Error) for each image
# A high error may indicate the image is "unfamiliar" (spoof Image)
errors = np.mean(np.square(all_images - reconstructed), axis=(1, 2, 3))

# Create a DataFrame to store errors and true labels for evaluation
df = pd.DataFrame({
    "reconstruction_error": errors,
    "true_label": all_labels
})
df.to_csv("reconstruction_errors.csv", index=False)
print("Reconstruction errors saved to 'reconstruction_errors.csv'")
print(df.head())

# Set the anomaly threshold
# Images with error > threshold are classified as spoof (1), else live (0)
threshold = 0.02  # This value can be tuned using ROC analysis

# Binary predictions based on the reconstruction error threshold
y_true = df["true_label"]
y_pred = (df["reconstruction_error"] > threshold).astype(int)

# --- Classification Report ---
# Provides precision, recall, F1-score for both classes
print("Classification Report:")
print(classification_report(y_true, y_pred))

# --- Confusion Matrix ---
# Visualize classification results with a heatmap
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# --- ROC Curve ---
# Measures model performance across all possible thresholds
fpr, tpr, _ = roc_curve(y_true, df["reconstruction_error"])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color='gray')
plt.title("ROC Curve (Autoencoder)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve_autoencoder.png")
plt.show()
