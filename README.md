# MNIST TensorFlow

MNIST digit classification using TensorFlow/Keras with a Convolutional Neural Network (CNN) architecture.

## Model Architecture

The classifier uses a simple CNN architecture:

- **Conv2D** (32 filters, 3x3 kernel) → ReLU → MaxPooling (2x2)
- **Conv2D** (64 filters, 3x3 kernel) → ReLU → MaxPooling (2x2)
- **Flatten** → Dense (128 units) → ReLU → Dropout (0.5) → Dense (10 units, softmax)

## Requirements

- Python 3.8+
- TensorFlow 2.10+
- NumPy 1.21+

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the training script:

```bash
python mnist_cnn.py
```

This will:
1. Download and preprocess the MNIST dataset
2. Create and train the CNN model for 10 epochs
3. Evaluate the model on the test set and print the accuracy

## Expected Results

The CNN model typically achieves **~99% accuracy** on the MNIST test set after 10 epochs of training.
