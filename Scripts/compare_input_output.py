import matplotlib.pyplot as plt
import tensorflow as tf
from load_limited_data import load_limited_dataset

# Loading test dataset (input = output) for evaluating the autoencoder
_, test_ds = load_limited_dataset()

# Loading the trained autoencoder model
autoencoder = tf.keras.models.load_model("autoencoder.keras", compile=False)

# Get one batch of test images
test_batch = next(iter(test_ds))

# Take the first 10 test images from the batch
originals = test_batch[0][:10]

# Use the autoencoder to reconstruct these images
reconstructed = autoencoder.predict(originals)

# --- Visualization: Compare Originals vs Reconstructions ---
fig, axes = plt.subplots(2, 10, figsize=(20, 4))

# Display original images (top row)
for i in range(10):
    axes[0, i].imshow(originals[i])
    axes[0, i].axis("off")

# Display reconstructed images (bottom row)
for i in range(10):
    axes[1, i].imshow(reconstructed[i])
    axes[1, i].axis("off")

# labels for rows
axes[0, 0].set_title("Original")
axes[1, 0].set_title("Reconstructed")

# Clean layout and save the figure
plt.tight_layout()
plt.savefig("compare_original_vs_reconstructed.png")
plt.show()

# Add overall title 
fig.suptitle("Original vs Reconstructed Images", fontsize=16)
