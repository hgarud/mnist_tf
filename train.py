"""Training script for MNIST CNN classifier."""

import tensorflow as tf
from tensorflow import keras

from model import create_cnn_model


def load_data():
    """Load and preprocess MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Add channel dimension
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    
    return (x_train, y_train), (x_test, y_test)


def train(epochs=10, batch_size=128):
    """Train the CNN model on MNIST."""
    # Load data
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # Create model
    model = create_cnn_model()
    model.summary()
    
    # Train
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f'\nTest accuracy: {test_acc:.4f}')
    print(f'Test loss: {test_loss:.4f}')
    
    # Save model
    model.save('mnist_cnn_model.keras')
    print('Model saved to mnist_cnn_model.keras')
    
    return model, history


if __name__ == '__main__':
    train()
