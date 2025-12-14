# MNIST TensorFlow

MNIST digit classification using TensorFlow/Keras with a ResNet using Preactivation Residual Blocks.

## Model Architecture

The classifier uses a ResNet with preactivation residual blocks (from "Identity Mappings in Deep Residual Networks"):

- **Initial Conv2D** (16 filters, 3x3)
- **Stage 1**: 2× PreactivationResBlock (16 filters)
- **Stage 2**: PreactivationResBlock (32 filters, stride=2) + PreactivationResBlock (32 filters)
- **Stage 3**: PreactivationResBlock (64 filters, stride=2) + PreactivationResBlock (64 filters)
- **Head**: BatchNorm → ReLU → GlobalAveragePooling → Dense (10, softmax)

### Preactivation ResBlock

Each preactivation residual block follows the pattern:
```
BN → ReLU → Conv → BN → ReLU → Conv + skip connection
```

## Learning Rate Schedule

- **Linear warmup**: 0 → 1e-5 over 5 epochs
- **Cosine decay**: 1e-5 → 0 over 15 epochs
- **Total**: 20 epochs

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Train the model:

```bash
python train.py
```

The trained model will be saved as `mnist_resnet_model.keras`.

## Requirements

- Python 3.8+
- TensorFlow 2.10+
