"""Model definition for the MNIST classifier."""

import tensorflow as tf


class PreActBlock(tf.keras.layers.Layer):
    """Pre-activation residual block.

    Uses the order: BatchNorm -> ReLU -> Conv (pre-activation)
    instead of Conv -> BatchNorm -> ReLU (post-activation).
    """

    def __init__(self, filters, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.strides = strides

    def build(self, input_shape):
        in_channels = input_shape[-1]

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(
            self.filters, 3, strides=self.strides, padding='same', use_bias=False
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(
            self.filters, 3, strides=1, padding='same', use_bias=False
        )

        if self.strides != 1 or in_channels != self.filters:
            self.shortcut = tf.keras.layers.Conv2D(
                self.filters, 1, strides=self.strides, padding='same', use_bias=False
            )
        else:
            self.shortcut = None

        super().build(input_shape)

    def call(self, x, training=None):
        out = self.bn1(x, training=training)
        out = tf.nn.relu(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(out)
        else:
            shortcut = x

        out = self.conv1(out)
        out = self.bn2(out, training=training)
        out = tf.nn.relu(out)
        out = self.conv2(out)

        return out + shortcut


def create_model() -> tf.keras.Model:
    """Create a pre-activation ResNet with 3 layers for MNIST classification."""
    inputs = tf.keras.Input(shape=(28, 28, 1))

    # Initial convolution
    x = tf.keras.layers.Conv2D(16, 3, strides=1, padding='same', use_bias=False)(inputs)

    # 3 pre-activation residual layers
    x = PreActBlock(16, strides=1)(x)
    x = PreActBlock(32, strides=2)(x)
    x = PreActBlock(64, strides=2)(x)

    # Final batch norm and activation before pooling
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Global average pooling and classification head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10)(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)
