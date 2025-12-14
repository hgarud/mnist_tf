"""Simple CNN model for MNIST classification."""

import tensorflow as tf
from tensorflow import keras
from keras import layers


def create_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """Create a simple CNN model for MNIST classification.
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        A compiled Keras model
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
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax'),
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
