"""
MNIST Digit Classification using a ResNet with Preactivation Blocks.

This module implements a Residual Network with preactivation blocks for
classifying handwritten digits from the MNIST dataset.
"""

import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class PreactivationResBlock(layers.Layer):
    """
    Preactivation Residual Block.

    Follows the pattern from "Identity Mappings in Deep Residual Networks":
    BN -> ReLU -> Conv -> BN -> ReLU -> Conv + skip connection
    """

    def __init__(self, filters, kernel_size=3, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.strides = strides

        self.bn1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(
            filters, kernel_size, strides=strides, padding='same', use_bias=False
        )
        self.bn2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(
            filters, kernel_size, strides=1, padding='same', use_bias=False
        )

        # Projection shortcut if dimensions change
        self.projection = None
        self.projection_bn = None

    def build(self, input_shape):
        input_filters = input_shape[-1]
        if self.strides != 1 or input_filters != self.filters:
            self.projection = layers.Conv2D(
                self.filters, 1, strides=self.strides, padding='same', use_bias=False
            )
            self.projection_bn = layers.BatchNormalization()
        super().build(input_shape)

    def call(self, inputs, training=None):
        # Preactivation path
        x = self.bn1(inputs, training=training)
        x = tf.nn.relu(x)

        # Shortcut from preactivated input
        if self.projection is not None:
            shortcut = self.projection(x)
            shortcut = self.projection_bn(shortcut, training=training)
        else:
            shortcut = inputs

        # First conv
        x = self.conv1(x)

        # Second preactivation + conv
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)

        return x + shortcut


class WarmupCosineDecaySchedule(keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule with linear warmup and cosine decay.

    Args:
        peak_lr: Peak learning rate after warmup.
        warmup_epochs: Number of epochs for linear warmup.
        decay_epochs: Number of epochs for cosine decay.
        steps_per_epoch: Number of training steps per epoch.
    """

    def __init__(self, peak_lr, warmup_epochs, decay_epochs, steps_per_epoch):
        super().__init__()
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.decay_steps = decay_epochs * steps_per_epoch
        self.total_steps = self.warmup_steps + self.decay_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        # Linear warmup phase: 0 -> peak_lr
        warmup_lr = (step / tf.maximum(self.warmup_steps, 1)) * self.peak_lr

        # Cosine decay phase: peak_lr -> 0
        decay_step = step - self.warmup_steps
        decay_progress = decay_step / tf.maximum(self.decay_steps, 1)
        cosine_decay = 0.5 * (1.0 + tf.cos(math.pi * decay_progress))
        decay_lr = self.peak_lr * cosine_decay

        # Select appropriate phase
        return tf.where(step < self.warmup_steps, warmup_lr, decay_lr)

    def get_config(self):
        return {
            'peak_lr': self.peak_lr,
            'warmup_steps': self.warmup_steps,
            'decay_steps': self.decay_steps,
            'total_steps': self.total_steps,
        }


def create_resnet_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Create a ResNet model with preactivation blocks for MNIST classification.

    Architecture:
        - Initial Conv2D (16 filters)
        - PreactivationResBlock (16 filters) x2
        - PreactivationResBlock (32 filters, stride=2) + PreactivationResBlock (32 filters)
        - PreactivationResBlock (64 filters, stride=2) + PreactivationResBlock (64 filters)
        - Global Average Pooling -> Dense (num_classes)

    Args:
        input_shape: Shape of input images (height, width, channels).
        num_classes: Number of output classes.

    Returns:
        An uncompiled Keras model.
    """
    inputs = keras.Input(shape=input_shape)

    # Initial convolution
    x = layers.Conv2D(16, 3, padding='same', use_bias=False)(inputs)

    # Stage 1: 16 filters
    x = PreactivationResBlock(16)(x)
    x = PreactivationResBlock(16)(x)

    # Stage 2: 32 filters, downsample
    x = PreactivationResBlock(32, strides=2)(x)
    x = PreactivationResBlock(32)(x)

    # Stage 3: 64 filters, downsample
    x = PreactivationResBlock(64, strides=2)(x)
    x = PreactivationResBlock(64)(x)

    # Final layers
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return keras.Model(inputs, outputs)


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
                warmup_epochs=5, decay_epochs=15, batch_size=128):
    """
    Train the ResNet model with warmup + cosine decay learning rate schedule.

    Args:
        model: Uncompiled Keras model.
        x_train: Training images.
        y_train: Training labels.
        x_test: Test images.
        y_test: Test labels.
        warmup_epochs: Number of epochs for linear LR warmup (0 -> peak_lr).
        decay_epochs: Number of epochs for cosine LR decay.
        batch_size: Batch size for training.

    Returns:
        Training history object.
    """
    total_epochs = warmup_epochs + decay_epochs

    # Calculate steps per epoch (accounting for validation split)
    train_samples = int(len(x_train) * 0.9)  # 10% validation split
    steps_per_epoch = train_samples // batch_size

    # Create learning rate schedule: linear warmup 0->1e-5, then cosine decay
    lr_schedule = WarmupCosineDecaySchedule(
        peak_lr=1e-5,
        warmup_epochs=warmup_epochs,
        decay_epochs=decay_epochs,
        steps_per_epoch=steps_per_epoch
    )

    # Compile model with scheduled learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=total_epochs,
        validation_split=0.1,
        verbose=1
    )

    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    return history


def main():
    """Main function to run MNIST ResNet training."""
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")

    print("\nCreating ResNet model with preactivation blocks...")
    model = create_resnet_model()
    model.summary()

    print("\nTraining with linear warmup (5 epochs) + cosine decay (15 epochs)...")
    print("Learning rate: 0 -> 1e-5 (warmup) -> 0 (cosine decay)")
    train_model(model, x_train, y_train, x_test, y_test)

    # Save the model
    model.save('mnist_resnet_model.keras')
    print("\nModel saved to 'mnist_resnet_model.keras'")


if __name__ == '__main__':
    main()
