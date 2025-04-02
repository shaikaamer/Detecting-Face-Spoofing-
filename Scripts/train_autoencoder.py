import tensorflow as tf
from tensorflow.keras import layers, models
from load_limited_data import load_limited_dataset
import matplotlib.pyplot as plt

# Loading unlabeled image dataset for reconstruction-based anomaly detection
# Autoencoder training
train_ds, test_ds = load_limited_dataset()

# Build a convolutional autoencoder model
autoencoder = models.Sequential([
    layers.Input(shape=(150, 150, 3)),

    # Encoder: progressively reducing spatial dimensions while increasing the depth
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2), padding='same'),
    
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2), padding='same'),
    
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2), padding='same'),

    # Decoder: mirror the encoder with upsampling to reconstruct the original input
    layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    
    layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    
    layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),

    # Final output layer with sigmoid to normalize pixel values [0, 1]
    layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same'),

    # Cropping ensures output shape matches original (150x150)
    layers.Cropping2D(((1, 1), (1, 1)))  # Remove border pixels to align dimensions
])

# Compile the model with Mean Squared Error loss, suitable for image reconstruction
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder and store history for visualization
history = autoencoder.fit(train_ds, validation_data=test_ds, epochs=20)

# Visualize training and validation loss curves
def plot_history(history, title="Model Training"):
    plt.figure(figsize=(10, 5))
    
    # Plot training and validation loss
    plt.plot(history.history['loss'], label="Train Loss")
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label="Val Loss")
    
    # plot accuracy
    if 'accuracy' in history.history:
        plt.plot(history.history['accuracy'], label="Train Accuracy")
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label="Val Accuracy")
    
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()

# Plot the training history
plot_history(history, title="Autoencoder Training")

# Save the training model
autoencoder.save("autoencoder.keras")
print("Autoencoder model saved to 'autoencoder.keras'")

# Print the model summary 
autoencoder.summary()
