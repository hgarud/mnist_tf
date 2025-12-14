# MNIST TensorFlow

MNIST digit classification using TensorFlow/Keras.

## Model Architecture

The classifier uses a simple CNN architecture:

- **Conv2D** (32 filters, 3x3 kernel, ReLU activation)
- **MaxPooling2D** (2x2 pool size)
- **Conv2D** (64 filters, 3x3 kernel, ReLU activation)
- **MaxPooling2D** (2x2 pool size)
- **Flatten**
- **Dropout** (0.5)
- **Dense** (128 units, ReLU activation)
- **Dense** (10 units, softmax activation)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Train the model:

```bash
python train.py --epochs 10 --batch-size 128
```

### Arguments

- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size for training (default: 128)
- `--save-path`: Path to save the trained model (default: mnist_cnn_model.keras)
