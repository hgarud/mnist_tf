"""Main training script for the MNIST classifier."""

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from dataset import get_datasets
from model import create_model


def main():
    """Main entrypoint."""

    ds_train, ds_test = get_datasets()

    model = create_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    model.fit(ds_train, epochs=5)

    model.evaluate(ds_test)

    model.save('model.keras')
    print("Model saved to model.keras")

    print("Model evaluated on test set")

if __name__ == "__main__":
    main()
