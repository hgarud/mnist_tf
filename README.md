# MNIST TensorFlow

MNIST digit classification using TensorFlow/Keras with a Convolutional Neural Network (CNN).

## Model Architecture

The classifier uses a simple CNN architecture:

- **Conv2D** (32 filters, 3x3 kernel) → ReLU → MaxPooling (2x2)
- **Conv2D** (64 filters, 3x3 kernel) → ReLU → MaxPooling (2x2)
- **Flatten** → Dense (128) → ReLU → Dropout (0.5) → Dense (10, softmax)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Train the model:

```bash
python train.py
```

The trained model will be saved as `mnist_cnn_model.keras`.

## Requirements

- Python 3.8+
- TensorFlow 2.10+
