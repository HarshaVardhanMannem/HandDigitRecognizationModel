# Training Results (MNIST)

This document summarizes the training outcomes from `MNIST-HandDigitRecognization.ipynb` and the saved model artifacts in `models/`.

## Model Artifacts

- `models/mnist_cnn_model.h5` — baseline CNN
- `models/mnist_cnn_model_augmented.h5` — augmented CNN (full run)
- `models/best_mnist_model.h5` — best checkpoint used by the Flask app

## Results Summary

- **Best model**: `best_mnist_model.h5`
- **Test accuracy**: **99.61%** (0.9961)
- **Macro precision/recall/F1**: **99.61%**
- **Macro average precision (AP)**: **99.99%**
- **Early stopping**: best validation accuracy at **epoch 11** (val_accuracy 0.9961)

## Training Pipeline (Notebook)

The training workflow in `MNIST-HandDigitRecognization.ipynb` follows these steps:

1. **Data preparation**
   - Load MNIST from Keras datasets
   - Normalize to [0, 1] and reshape to (28, 28, 1)
   - Split into training and validation sets
2. **Baseline CNN**
   - Train a simple CNN without augmentation
   - Save to `models/mnist_cnn_model.h5`
3. **Augmented CNN**
   - Apply rotation, shift, zoom, and shear augmentation
   - Add Gaussian noise, batch normalization, and dropout
   - Train with callbacks (ReduceLROnPlateau + EarlyStopping)
   - Save full model to `models/mnist_cnn_model_augmented.h5`
   - Save best checkpoint as `models/best_mnist_model.h5`
4. **Evaluation**
   - Report accuracy, precision/recall/F1, and confusion matrix
   - Generate precision–recall curves for class-level insight

## Baseline vs Augmented

| Model | Key Differences | Best Val Accuracy | Test Accuracy | Artifact |
|-------|-----------------|-------------------|---------------|----------|
| Baseline CNN | No augmentation, simpler training loop | **99.00%** (epoch 5) | — | `mnist_cnn_model.h5` |
| Augmented CNN | Data augmentation + regularization + callbacks | **99.61%** (epoch 11) | **99.61%** | `best_mnist_model.h5` |

## Training Highlights (Augmented Model)

| Epoch | Train Accuracy | Val Accuracy | Val Loss | Learning Rate |
|-------|---------------|--------------|----------|---------------|
| 1     | 89.49%        | 86.78%       | 0.3314   | 1e-3          |
| 2     | 96.67%        | 98.99%       | 0.0308   | 1e-3          |
| 4     | 97.86%        | 99.25%       | 0.0230   | 1e-3          |
| 8     | 98.69%        | 99.53%       | 0.0141   | 5e-4          |
| 11    | 98.90%        | **99.61%**   | 0.0109   | 5e-4 ✓ best  |
| 16    | 99.10%        | 99.55%       | 0.0112   | 2.5e-4 (stop) |
