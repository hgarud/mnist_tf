"""Model definition for the MNIST classifier."""

import tensorflow as tf


class PreActivationResidualBlock(tf.keras.layers.Layer):
    """Pre-activation residual block: BN -> ReLU -> Conv -> BN -> ReLU -> Conv."""

    def __init__(self, filters, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.strides = strides

    def build(self, input_shape):
        in_filters = input_shape[-1]

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(
            self.filters, (3, 3), strides=self.strides, padding='same', use_bias=False
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(
            self.filters, (3, 3), strides=1, padding='same', use_bias=False
        )

        # Projection shortcut if dimensions change
        if self.strides != 1 or in_filters != self.filters:
            self.shortcut = tf.keras.layers.Conv2D(
                self.filters, (1, 1), strides=self.strides, padding='same', use_bias=False
            )
        else:
            self.shortcut = None

        super().build(input_shape)

    def call(self, inputs, training=None):
        x = self.bn1(inputs, training=training)
        x = tf.nn.relu(x)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = inputs

        x = self.conv1(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)

        return x + shortcut


def create_model() -> tf.keras.Model:
    """Create a pre-activation ResNet for MNIST with 3 convolutional layers."""
    inputs = tf.keras.Input(shape=(28, 28, 1))

    # Initial convolution (layer 1)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', use_bias=False)(inputs)

    # Pre-activation residual block with 2 conv layers (layers 2 and 3)
    # Using stride=2 to downsample (similar to original MaxPooling)
    x = PreActivationResidualBlock(64, strides=2)(x)

    # Final batch norm and activation before classification
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Classification head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(10)(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)
