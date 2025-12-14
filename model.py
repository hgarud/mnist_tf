"""Simple CNN model for MNIST digit classification."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """Create a simple CNN model for MNIST classification.
    
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
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


class MNISTClassifier:
    """MNIST digit classifier using a simple CNN architecture."""
    
    def __init__(self):
        self.model = create_cnn_model()
        
    def train(self, x_train, y_train, x_val=None, y_val=None, 
              epochs=10, batch_size=128, callbacks=None):
        """Train the model.
        
        Args:
            x_train: Training images.
            y_train: Training labels.
            x_val: Validation images.
            y_val: Validation labels.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            callbacks: Optional list of Keras callbacks.
            
        Returns:
            Training history.
        """
        validation_data = None
        if x_val is not None and y_val is not None:
            validation_data = (x_val, y_val)
            
        history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks
        )
        return history
    
    def evaluate(self, x_test, y_test):
        """Evaluate the model on test data.
        
        Args:
            x_test: Test images.
            y_test: Test labels.
            
        Returns:
            Tuple of (loss, accuracy).
        """
        return self.model.evaluate(x_test, y_test)
    
    def predict(self, x):
        """Make predictions on input data.
        
        Args:
            x: Input images.
            
        Returns:
            Predicted class probabilities.
        """
        return self.model.predict(x)
    
    def save(self, filepath):
        """Save the model to a file.
        
        Args:
            filepath: Path to save the model.
        """
        self.model.save(filepath)
        
    def load(self, filepath):
        """Load a model from a file.
        
        Args:
            filepath: Path to the saved model.
        """
        self.model = keras.models.load_model(filepath)
    
    def summary(self):
        """Print model summary."""
        self.model.summary()
