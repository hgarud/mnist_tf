"""Model definition for the MNIST classifier."""

import tensorflow as tf


def residual_block(x, filters, kernel_size=3, stride=1, downsample=False):
    """A single residual block with skip connection."""
    shortcut = x

    # First convolution
    x = tf.keras.layers.Conv2D(
        filters, kernel_size, strides=stride, padding='same', use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Second convolution
    x = tf.keras.layers.Conv2D(
        filters, kernel_size, strides=1, padding='same', use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Adjust shortcut if dimensions change
    if downsample or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(
            filters, 1, strides=stride, padding='same', use_bias=False
        )(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    # Add skip connection
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)

    return x


def create_model() -> tf.keras.Model:
    """Create a ResNet model with 4 residual layers for MNIST classification."""
    inputs = tf.keras.layers.Input(shape=(28, 28))

    # Reshape to add channel dimension (28, 28) -> (28, 28, 1)
    x = tf.keras.layers.Reshape((28, 28, 1))(inputs)

    # Initial convolution
    x = tf.keras.layers.Conv2D(32, 3, strides=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # 4 Residual layers
    x = residual_block(x, filters=32)   # Layer 1: 28x28x32
    x = residual_block(x, filters=64, stride=2, downsample=True)   # Layer 2: 14x14x64
    x = residual_block(x, filters=128, stride=2, downsample=True)  # Layer 3: 7x7x128
    x = residual_block(x, filters=256, stride=2, downsample=True)  # Layer 4: 4x4x256

    # Global average pooling and classification head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
