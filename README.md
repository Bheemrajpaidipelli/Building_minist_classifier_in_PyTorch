# Building_minist_classifier_in_PyTorch

A simple but powerful feed-forward neural network (MLP) for classifying handwritten digits from the [MNIST dataset] implemented in **PyTorch**.

## Project Overview

This project demonstrates:
- Loading and pre-processing MNIST data using `torchvision`.
- Building a fully-connected neural network classifier using `PyTorch nn.Module`.
- Training the model with different optimizers and hyperparameters.
- Evaluating accuracy and loss on validation data.
- Saving trained model weights.

## Results

| Optimizer         | Epochs | Momentum | Nesterov | Test Accuracy | Test Loss |
|-------------------|--------|----------|----------|--------------|-----------|
| SGD               | 20     | No       | No       | 89.46%       | 0.4034    |
| SGD               | 50     | No       | No       | 91.00%       | 0.3007    |
| SGD               | 20     | 0.9      | No       | 98.05%       | 0.0790    |
| SGD               | 20     | 0.9      | Yes      | 97.91%       | 0.0647    |
| Adam              | 20     | -        | -        | 98.10%       | 0.0807    |

Best test accuracy: **98.10%** using Adam optimizer.

## Requirements

- Python 3.7+
- torch
- torchvision
- matplotlib

Install dependencies via pip:

```bash
pip install torch torchvision matplotlib
```

## Files

- `Building-MINIST-classifier-in-PyTorch.ipynb`: Main Jupyter notebook.
- `mnist.pt`: Trained model weights (created after training).

## Usage

### 1. Clone the repository and navigate to the project folder
```bash
git clone 
cd 
```

### 2. Train the model

Open the notebook in Jupyter:

```bash
jupyter notebook Building-MINIST-classifier-in-PyTorch.ipynb
```

Run all code cells to:
- Load MNIST data
- Define the model and optimizer (`Adam` by default)
- Train for 20 epochs
- Plot training loss
- Save the model as `mnist.pt`

### 3. Evaluate the model

Model is evaluated on the MNIST test dataset. Final output will show test loss and accuracy, for example:

```
Test loss: 0.0807, Test accuracy: 98.10%

- **Input:** 784 (28×28 pixels flattened)
- **Hidden:** 256 units, ReLU activation
- **Output:** 10 units (one per digit 0–9)

## Hyperparameter Tuning and Optimizer Notes

- Adding **momentum** to SGD greatly improves accuracy.
- **Nesterov Accelerated Gradient** (nesterov=True) provides a slight improvement over standard momentum.
- The **Adam** optimizer gives fast and robust convergence with default parameters.
- Training for more epochs can modestly improve accuracy for vanilla SGD.

## Additional Notes

- `super()` is used in the custom model to properly initialize `nn.Module`.
- Model weights are saved as `mnist.pt` after training.
- All data loading is handled with `torchvision.datasets.MNIST`.

## References

- [PyTorch documentation](https://pytorch.org/docs/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

**Author:** Bheemraj
**Date:** 04/08/2025
