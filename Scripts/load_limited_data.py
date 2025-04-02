from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_limited_dataset():
    """
    Loads a dataset for unsupervised training (autoencoder) using data augmentation.
    
    Returns:
        train_ds,test_ds
    """

    # Initialize ImageDataGenerator with augmentation for generalization
    # - Rescaling normalizes pixel values to [0, 1] range
    # - Horizontal flipping helps model generalize across left/right face variations
    # - Rotation and zoom add diversity, simulating real-world noise and variability
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        horizontal_flip=True,
        rotation_range=20,
        zoom_range=0.2
    )

    # Load training dataset:
    # - Images are loaded from 'Data_limited/train'
    # - Resized to 150x150 to match autoencoder input shape
    # - class_mode='input' ensures images are returned as (input, input) pairs for reconstruction
    train_ds = datagen.flow_from_directory(
        "Data_limited/train",
        target_size=(150, 150),
        batch_size=32,
        class_mode="input"
    )

    # Load testing dataset similarly for evaluation
    test_ds = datagen.flow_from_directory(
        "Data_limited/test",
        target_size=(150, 150),
        batch_size=32,
        class_mode="input"
    )

    # Return both training and test dataset
    return train_ds, test_ds
