# Handwritten Digit Recognition

A neural network-based solution for handwritten digit recognition using TensorFlow and Streamlit. The project includes both the training pipeline and an interactive web application for recognizing uploaded MNIST-format digit images.

## 🌟 Features

- **CNN Model**: Convolutional Neural Network optimized for digit recognition
- **Upload Interface**: Upload MNIST-format images for instant recognition
- **High Accuracy**: ~99.3% test accuracy on the MNIST dataset
- **Visualization**: See confidence scores and probability distributions
- **Data Augmentation**: Improved model generalization through augmentation techniques

## 📋 Project Structure

```
mnist-digit-recognition/
├── models/                   # Directory for saved models
│   └── mnist_cnn_best.h5     # Best model checkpoint during training
├── mnist_cnn_model.h5        # Final trained model
├── train_model.py            # CNN training script
├── app.py                    # Streamlit web application
└── README.md                 # Project documentation
```

## 🧠 Neural Network Architecture

The CNN architecture consists of:

1. **Three Convolutional Blocks**:
   - Each block contains two Conv2D layers with increasing filter sizes (32→64→128)
   - Batch normalization for stable training
   - MaxPooling for spatial dimension reduction
   - Dropout for regularization

2. **Dense Layers**:
   - 256 neurons with ReLU activation and L2 regularization
   - Output layer with 10 neurons (one per digit) and softmax activation
   
3. **Training Configuration**:
   - Adam optimizer with learning rate scheduling
   - Early stopping to prevent overfitting
   - Data augmentation with rotation, shifting, zooming

## 💻 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mnist-digit-recognition.git
   cd mnist-digit-recognition
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Requirements:
   ```
   tensorflow>=2.8.0
   numpy>=1.20.0
   matplotlib>=3.5.0
   scikit-learn>=1.0.0
   streamlit>=1.18.0
   pillow>=9.0.0
   ```

## 🚀 Usage

### Training the Model

Run the training script to train the CNN model on the MNIST dataset:

```bash
python train_model.py
```

This script will:
- Download the MNIST dataset
- Preprocess the data
- Train the CNN model
- Evaluate model performance
- Save the trained model as `mnist_cnn_model.h5`

### Running the Web Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

The application provides the following functionality:
- **Upload Image**: Upload an image containing a handwritten digit in MNIST format (28x28 pixels, grayscale)

## 📊 Performance

The model achieves excellent performance on the MNIST dataset:
- Training accuracy: ~99.5%
- Validation accuracy: ~99.4%
- Test accuracy: ~99.3%

## 🌐 Web Application Details

The Streamlit web application offers a user-friendly interface with:

- **Image Upload**: Test with existing images of handwritten digits in MNIST format
- **Preprocessing Visualization**: See how images are processed before prediction
- **Confidence Scores**: Visualize model's confidence for each digit class
- **Top Predictions**: View top 3 most likely digits and their probabilities

## 🔍 Future Improvements

- Implement batch prediction for multiple digits
- Add model interpretability tools (like Grad-CAM)
- Support for handwritten character recognition beyond digits

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

HARSHA VARDHAN MANNEM
