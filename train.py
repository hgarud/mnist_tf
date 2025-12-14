"""Training script for MNIST CNN classifier."""

import argparse
import tensorflow as tf
from tensorflow import keras

from model import MNISTClassifier


def load_mnist_data():
    """Load and preprocess MNIST dataset.
    
    Returns:
        Tuple of (x_train, y_train, x_test, y_test).
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape to add channel dimension (28, 28) -> (28, 28, 1)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    return x_train, y_train, x_test, y_test


def main():
    parser = argparse.ArgumentParser(description='Train MNIST CNN classifier')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--save-path', type=str, default='mnist_cnn_model.keras', 
                        help='Path to save the trained model')
    args = parser.parse_args()
    
    print("Loading MNIST dataset...")
    x_train, y_train, x_test, y_test = load_mnist_data()
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    
    # Split training data for validation
    val_split = 0.1
    val_size = int(len(x_train) * val_split)
    x_val, y_val = x_train[:val_size], y_train[:val_size]
    x_train, y_train = x_train[val_size:], y_train[val_size:]
    
    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Test samples: {len(x_test)}")
    
    # Create and train model
    print("\nCreating CNN model...")
    classifier = MNISTClassifier()
    classifier.summary()
    
    print(f"\nTraining for {args.epochs} epochs...")
    history = classifier.train(
        x_train, y_train,
        x_val=x_val, y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = classifier.evaluate(x_test, y_test)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Save model
    print(f"\nSaving model to {args.save_path}...")
    classifier.save(args.save_path)
    print("Done!")


if __name__ == '__main__':
    main()
