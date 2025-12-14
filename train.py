"""
MNIST Digit Classification using a Simple CNN.

This module implements a Convolutional Neural Network (CNN) for classifying
handwritten digits from the MNIST dataset.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Create a simple CNN model for MNIST digit classification.

    Architecture:
        - Conv2D (32 filters, 3x3) -> ReLU -> MaxPooling (2x2)
        - Conv2D (64 filters, 3x3) -> ReLU -> MaxPooling (2x2)
        - Flatten -> Dense (128) -> ReLU -> Dropout -> Dense (num_classes)

    Args:
        input_shape: Shape of input images (height, width, channels).
        num_classes: Number of output classes.

    Returns:
        A compiled Keras model.
    """
    model = keras.Sequential([
        # First convolutional block
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', 
                      input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Second convolutional block
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def load_and_preprocess_data():
    """
    Load and preprocess the MNIST dataset.

    Returns:
        Tuple of (x_train, y_train), (x_test, y_test) with normalized pixel values.
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Add channel dimension (28, 28) -> (28, 28, 1)
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    return (x_train, y_train), (x_test, y_test)


def train_model(model, x_train, y_train, x_test, y_test, 
                epochs=10, batch_size=128):
    """
    Train the CNN model on MNIST data.

    Args:
        model: Compiled Keras model.
        x_train: Training images.
        y_train: Training labels.
        x_test: Test images.
        y_test: Test labels.
        epochs: Number of training epochs.
        batch_size: Batch size for training.

    Returns:
        Training history object.
    """
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        verbose=1
    )

    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    return history


def main():
    """Main function to run MNIST CNN training."""
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")

    print("\nCreating CNN model...")
    model = create_cnn_model()
    model.summary()

    print("\nTraining model...")
    train_model(model, x_train, y_train, x_test, y_test)

    # Save the model
    model.save('mnist_cnn_model.keras')
    print("\nModel saved to 'mnist_cnn_model.keras'")


if __name__ == '__main__':
    main()
