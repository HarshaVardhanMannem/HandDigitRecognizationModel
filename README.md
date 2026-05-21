# Handwritten Digit Recognition (MNIST)

An end-to-end handwritten digit recognition project that pairs a TensorFlow CNN training notebook with a Flask web app for real-time canvas inference. It ships with trained model artifacts and documented evaluation results so contributors can reproduce, improve, and deploy the model quickly.

## 🌟 Highlights

- **Two CNN baselines**: a simple CNN and a deeper, augmented CNN with regularization
- **Interactive web UI**: draw digits in the browser, preview preprocessing, and view top-3 predictions
- **Reproducible training**: notebook-driven pipeline that saves checkpoints and evaluation artifacts
- **Strong accuracy**: 99.61% test accuracy for the best augmented model

## 📦 Dataset

This project trains on the **MNIST** dataset:
- **60,000** training images and **10,000** test images
- 28×28 grayscale digits (0–9)
- Normalized to [0, 1] and reshaped to (28, 28, 1)

## 📋 Project Structure

```
HandDigitRecognizationModel/
├── models/
│   ├── best_mnist_model.h5           # Best model checkpoint (used by the web app)
│   ├── mnist_cnn_model.h5            # Basic CNN model
│   └── mnist_cnn_model_augmented.h5  # Augmented CNN model
├── static/                           # CSS and JavaScript assets
├── templates/
│   └── index.html                    # Web app front-end
├── MNIST-HandDigitRecognization.ipynb # Training notebook
├── app.py                            # Flask web application
└── README.md                         # Project documentation
```

## 🧠 Model Architecture (best_mnist_model.h5)

The production model is a deeper CNN trained with augmentation and regularization:

1. **Convolutional Blocks**
   - Block 1: Two Conv2D layers (32 filters, 3×3) + BatchNorm + MaxPooling + Dropout(0.25)
   - Block 2: Two Conv2D layers (64 filters, 3×3) + BatchNorm + MaxPooling + Dropout(0.25)
   - Input noise layer (GaussianNoise σ=0.1) for robustness

2. **Dense Head**
   - 256 neurons with ReLU activation + BatchNorm + Dropout(0.5)
   - Output layer with 10 neurons (softmax)

3. **Training Configuration**
   - Adam optimizer with ReduceLROnPlateau
   - Early stopping (patience=5, restore best weights)
   - Data augmentation: rotation ±10°, width/height shift ±10%, zoom ±10%, shear ±10%
   - Batch size: 128 | Max epochs: 30 (stopped at epoch 16)

4. **Total Parameters**: 871,530 trainable (≈3.32 MB)

## 🔬 Training Pipeline (Notebook)

The notebook `MNIST-HandDigitRecognization.ipynb` walks through:

1. **Data prep**: load MNIST, normalize to [0, 1], reshape to (28, 28, 1)
2. **Baseline CNN**: trains `mnist_cnn_model.h5` (simple CNN without augmentation)
3. **Augmented CNN**: trains `mnist_cnn_model_augmented.h5` with callbacks and saves `best_mnist_model.h5`
4. **Evaluation**: confusion matrix, classification report, and precision–recall curves

## 🔎 Inference Pipeline (Flask App)

The web app (`app.py`) uses the same preprocessing as training:

1. Capture canvas input and send as base64 PNG
2. Convert to grayscale, resize to 28×28, invert, and normalize
3. Run inference, return top-1 prediction, confidence, top-3 list, and a processed-image preview

## 💻 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/HarshaVardhanMannem/HandDigitRecognizationModel.git
   cd HandDigitRecognizationModel
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install tensorflow>=2.8.0 numpy>=1.20.0 matplotlib>=3.5.0 scikit-learn>=1.0.0 flask>=2.0.0 pillow>=9.0.0
   ```

3. Optional (for notebook execution):
   ```bash
   pip install jupyter
   ```

## 🚀 Usage

### Training the Model

Open and run the Jupyter notebook to train the CNN model on the MNIST dataset:

```bash
jupyter notebook MNIST-HandDigitRecognization.ipynb
```

The notebook will:
- Download and preprocess the MNIST dataset
- Train a basic CNN model and save it as `models/mnist_cnn_model.h5`
- Train an augmented CNN model with callbacks and save the best checkpoint as `models/best_mnist_model.h5`
- Evaluate model performance and display confusion matrix, classification report, and precision–recall curves

### Running the Web Application

Launch the Flask application:

```bash
python app.py
```

Then open your browser at `http://127.0.0.1:5000`. The application provides:
- **Interactive Canvas**: Draw a digit directly in the browser
- **Preprocessing Visualization**: See how your drawing is processed before prediction
- **Confidence Scores**: View the model's confidence for each digit class
- **Top 3 Predictions**: The three most likely digits and their probabilities

## 📊 Results & Improvements

### Baseline vs Augmented (from notebook runs)

| Model | Key Differences | Best Val Accuracy | Test Accuracy | Artifact |
|-------|-----------------|-------------------|---------------|----------|
| Baseline CNN | No augmentation, simpler training loop | **99.00%** (epoch 5) | — | `mnist_cnn_model.h5` |
| Augmented CNN | Data augmentation + regularization + callbacks | **99.61%** (epoch 11) | **99.61%** | `best_mnist_model.h5` |

### Why the Augmented Model Improves

- **Data augmentation** expands the effective dataset, improving generalization to new handwriting styles
- **Gaussian noise + dropout** reduce overfitting to training strokes
- **Batch normalization** stabilizes optimization and speeds convergence
- **ReduceLROnPlateau + early stopping** preserve the best-performing checkpoint

### Training History (Augmented Model — best_mnist_model.h5)

### Training History (Augmented Model — best_mnist_model.h5)

| Epoch | Train Accuracy | Val Accuracy | Val Loss | Learning Rate |
|-------|---------------|--------------|----------|---------------|
| 1     | 89.49%        | 86.78%       | 0.3314   | 1e-3          |
| 2     | 96.67%        | 98.99%       | 0.0308   | 1e-3          |
| 4     | 97.86%        | 99.25%       | 0.0230   | 1e-3          |
| 8     | 98.69%        | 99.53%       | 0.0141   | 5e-4          |
| 11    | 98.90%        | **99.61%**   | 0.0109   | 5e-4 ✓ best  |
| 16    | 99.10%        | 99.55%       | 0.0112   | 2.5e-4 (stop) |

*Early stopping restored weights from epoch 11 (best val_accuracy = 99.61%).*

### Test Set Performance (10,000 samples)

| Metric   | Score  |
|----------|--------|
| Accuracy | **99.61%** |
| Macro Precision | 99.61% |
| Macro Recall    | 99.61% |
| Macro F1-Score  | 99.61% |
| Macro Average Precision (AP) | **99.99%** |

### Per-Class Classification Report

| Digit | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 99.59%    | 99.90% | 99.75%   | 980     |
| 1     | 99.74%    | 99.65% | 99.69%   | 1135    |
| 2     | 99.61%    | 99.81% | 99.71%   | 1032    |
| 3     | 99.51%    | 99.80% | 99.65%   | 1010    |
| 4     | 99.29%    | 99.69% | 99.49%   | 982     |
| 5     | 99.78%    | 99.44% | 99.61%   | 892     |
| 6     | 99.89%    | 99.16% | 99.53%   | 958     |
| 7     | 99.42%    | 99.81% | 99.61%   | 1028    |
| 8     | 99.59%    | 99.59% | 99.59%   | 974     |
| 9     | 99.70%    | 99.21% | 99.45%   | 1009    |

## 🤝 Contributing

Contributions are welcome! If you improve the model, the UI, or documentation:

1. Fork the repository
2. Create a feature branch
3. Commit changes with clear messages
4. Open a pull request with a short summary and results

## 🔍 Future Improvements

- Implement batch prediction for multiple digits
- Add model interpretability tools (like Grad-CAM)
- Support for handwritten character recognition beyond digits

## 📄 License

No license file is included yet. Add a LICENSE file to clarify usage and distribution.

## 👨‍💻 Author

HARSHA VARDHAN MANNEM
