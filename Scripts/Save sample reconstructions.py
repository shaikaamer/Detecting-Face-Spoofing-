import cv2
import os
import tensorflow as tf
import numpy as np
from load_limited_data import load_limited_dataset

# Load the trained autoencoder model
autoencoder = tf.keras.models.load_model("autoencoder.keras", compile=False)

# Load test dataset (input = output) for reconstruction comparison
_, test_ds = load_limited_dataset()

# Directory to save the output images
output_dir = "reconstructed_outputs"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Fetch a batch of test images
batch = next(iter(test_ds))

# Select the first 10 images for reconstruction
originals = batch[0][:10]

# Reconstruct the images using the autoencoder
reconstructions = autoencoder.predict(originals)

# Save both original and reconstructed images side-by-side
for i in range(10):
    # Rescale pixel values from [0, 1] back to [0, 255] for saving
    orig = (originals[i] * 255).astype(np.uint8)
    recon = (reconstructions[i] * 255).astype(np.uint8)

    # Convert from RGB to BGR for OpenCV compatibility
    cv2.imwrite(f"{output_dir}/original_{i}.png", cv2.cvtColor(orig, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{output_dir}/reconstructed_{i}.png", cv2.cvtColor(recon, cv2.COLOR_RGB2BGR))
