"""
MNIST Digit Classification using a Simple CNN Architecture.

This module implements a Convolutional Neural Network (CNN) for classifying
handwritten digits from the MNIST dataset using TensorFlow/Keras.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def create_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Create a simple CNN model for MNIST digit classification.
    
    Architecture:
        - Conv2D (32 filters, 3x3) -> ReLU -> MaxPool (2x2)
        - Conv2D (64 filters, 3x3) -> ReLU -> MaxPool (2x2)
        - Flatten -> Dense (128) -> ReLU -> Dropout -> Dense (num_classes)
    
    Args:
        input_shape: Shape of input images (height, width, channels).
        num_classes: Number of output classes.
    
    Returns:
        A compiled Keras model.
    """
    model = keras.Sequential([
        # First convolutional block
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Second convolutional block
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def load_and_preprocess_data():
    """
    Load and preprocess the MNIST dataset.
    
    Returns:
        Tuple of (x_train, y_train, x_test, y_test) with preprocessed data.
    """
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape to add channel dimension (28, 28) -> (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    # Convert labels to one-hot encoding
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    return x_train, y_train, x_test, y_test


def train_model(model, x_train, y_train, x_test, y_test, epochs=10, batch_size=128):
    """
    Train the CNN model on MNIST data.
    
    Args:
        model: Compiled Keras model.
        x_train: Training images.
        y_train: Training labels (one-hot encoded).
        x_test: Test images.
        y_test: Test labels (one-hot encoded).
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
    
    return history


def evaluate_model(model, x_test, y_test):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained Keras model.
        x_test: Test images.
        y_test: Test labels (one-hot encoded).
    
    Returns:
        Tuple of (test_loss, test_accuracy).
    """
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    return test_loss, test_accuracy


def main():
    """Main function to train and evaluate the MNIST CNN classifier."""
    print("Loading and preprocessing MNIST data...")
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    
    print("\nCreating CNN model...")
    model = create_cnn_model()
    model.summary()
    
    print("\nTraining model...")
    history = train_model(model, x_train, y_train, x_test, y_test)
    
    print("\nEvaluating model...")
    test_loss, test_accuracy = evaluate_model(model, x_test, y_test)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    return model, history


if __name__ == "__main__":
    main()
