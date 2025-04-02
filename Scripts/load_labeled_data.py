from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_labeled_dataset():
    """
    Loads the training and test datasets for binary classification (e.g., live vs. spoof).
    Applies basic augmentation to improve generalization of the classifier.

    Returns:
        train_ds: Training dataset with binary labels.
        test_ds: Test dataset with binary labels.
    """

    # Initialize ImageDataGenerator with preprocessing and augmentation
    # - Rescaling normalizes pixel values from [0, 255] to [0, 1]
    # - Horizontal flip helps generalize left/right face variations
    # - Rotation and zoom simulate real-world distortions, improving robustness
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        horizontal_flip=True,
        rotation_range=20,
        zoom_range=0.2
    )

    # Load training dataset with binary labels
    # - Directory structure must be: /train/class_0/ and /train/class_1/
    # - Target size matches the input shape expected by the model
    # - class_mode='binary' returns integer labels: 0 or 1
    train_ds = datagen.flow_from_directory(
        "Data_limited/train",
        target_size=(150, 150),
        batch_size=32,
        class_mode="binary"
    )

    # Load test dataset similarly
    test_ds = datagen.flow_from_directory(
        "Data_limited/test",
        target_size=(150, 150),
        batch_size=32,
        class_mode="binary"
    )

    # Return both datasets for use in training and evaluation
    return train_ds, test_ds
