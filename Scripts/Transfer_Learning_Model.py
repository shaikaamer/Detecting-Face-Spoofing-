import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from load_labeled_data import load_labeled_dataset

# Loading binary-labeled training and test datasets
# Classes: Live = 0, Spoof = 1
train_ds, test_ds = load_labeled_dataset()

# Loading MobileNetV2
# - Pretrained ImageNet weights for feature extractions
# - input_shape should match our resized image
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False  

# Binary classification
# - GlobalAveragePooling2D reduces dimensions while keeping important spatial features
# - Dense(128) introduces non-linearity and learns from extracted features
# - Dense(1, sigmoid) outputs a probability for binary classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
output = Dense(1, activation="sigmoid")(x)

# Combine base model and custom head into a single model
classifier = Model(inputs=base_model.input, outputs=output)

# Fine-tune last 10 layers of the base model to adapt pretrained weights to our data
for layer in base_model.layers[-10:]:
    layer.trainable = True

# Compile the model 
classifier.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Training the model and storing the training-history
history = classifier.fit(train_ds, validation_data=test_ds, epochs=10)

# Plot training and validation curves for both loss and accuracy
def plot_history(history, title="Classifier Training"):
    plt.figure(figsize=(10, 5))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label="Train Loss")
    plt.plot(history.history['val_loss'], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label="Train Accuracy")
    plt.plot(history.history['val_accuracy'], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True)

    # Format and save plot
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig("classifier_training.png")
    plt.show()

# Display training performance
plot_history(history)

# Save the trained model 
classifier.save("anomaly_classifier.keras")
print("✅ Model saved as 'anomaly_classifier.keras'")

# Model summary
classifier.summary()
