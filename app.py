import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import base64
from io import BytesIO
from streamlit_drawable_canvas import st_canvas
import pandas as pd
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Digit Recognition App",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        padding: 1rem;
        background: linear-gradient(to right, #E3F2FD, #BBDEFB, #E3F2FD);
        border-radius: 10px;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        color: #2C3E50;
    }
    .prediction-text {
        font-size: 5rem;
        font-weight: 800;
        text-align: center;
        color: #4CAF50;
        margin: 1.5rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        animation: fadeIn 0.5s ease-in;
    }
    .confidence-text {
        font-size: 1.4rem;
        text-align: center;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .canvas-container {
        display: flex;
        justify-content: center;
        margin: 1.5rem 0;
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .canvas-instructions {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin: 1rem 0;
        padding: 1rem;
        background: #E3F2FD;
        border-radius: 8px;
        border-left: 4px solid #1E88E5;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        margin: 0.5rem 0;
    }
    .predict-button>button {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    .predict-button>button:hover {
        background-color: #45a049 !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .clear-button>button {
        background-color: #f44336 !important;
        color: white !important;
    }
    .clear-button>button:hover {
        background-color: #da190b !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .result-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 1rem;
    }
    .preprocessed-image {
        display: flex;
        justify-content: center;
        align-items: center;
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained neural network model
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        with st.spinner('Loading model... This might take a moment.'):
            return tf.keras.models.load_model('mnist_cnn_model.h5')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Function to preprocess the image
@st.cache_data
def preprocess_image(image, invert=True):
    """Preprocess the image for neural network prediction."""
    try:
        # Convert to grayscale
        image = image.convert('L')
        
        # Resize to 28x28 (MNIST format)
        image = image.resize((28, 28), Image.LANCZOS)
        
        # Invert colors if needed (MNIST expects white digits on black background)
        if invert:
            image = ImageOps.invert(image)
            
        # Convert to numpy array and normalize
        img_array = np.array(image).astype('float32') / 255.0
        
        # Reshape for CNN input (batch_size, height, width, channels)
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

# Function to get predictions and visualization
def get_prediction_data(img_array, model):
    # Make prediction
    prediction = model.predict(img_array, verbose=0)
    
    # Get the predicted digit
    predicted_digit = np.argmax(prediction[0])
    confidence = float(prediction[0][predicted_digit])
    
    # Get top 3 predictions
    top3_indices = np.argsort(prediction[0])[-3:][::-1]
    top3_values = [float(prediction[0][i]) for i in top3_indices]
    top3 = [{"digit": int(i), "confidence": v} for i, v in zip(top3_indices, top3_values)]
    
    return {
        'prediction': predicted_digit,
        'confidence': confidence,
        'probabilities': prediction[0],
        'top3': top3
    }

# Function to create probability bar chart
def create_probability_chart(probabilities):
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Bar chart with custom styling
    bars = ax.bar(
        range(10), 
        probabilities, 
        color=['#1976D2' if i == np.argmax(probabilities) else '#90CAF9' for i in range(10)]
    )
    
    # Add labels and formatting
    ax.set_xlabel('Digit', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_xticks(range(10))
    ax.set_xticklabels(range(10), fontsize=10)
    ax.set_title('Probability Distribution', fontsize=14)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        if height > 0.01:  # Only show labels for significant probabilities
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01,
                f'{height:.2f}',
                ha='center', 
                fontsize=9
            )
    
    plt.tight_layout()
    return fig

# Main function
def main():
    # Header
    st.markdown("<div class='main-header'>✏️ Handwritten Digit Recognition</div>", unsafe_allow_html=True)
    st.markdown("Draw a digit (0-9) on the canvas below and see the neural network prediction!")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.warning("Failed to load the model. Please check if 'mnist_cnn_model.h5' exists.")
        return
    
    # Create tabs
    tab1, tab2 = st.tabs(["🎨 Draw Digit", "ℹ️ About"])
    
    # Tab 1: Draw Digit
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Handle canvas clearing
            canvas_key = "canvas_" + str(hash(datetime.now().isoformat())) if st.session_state.get("should_clear_canvas", False) else "canvas"
            
            if st.session_state.get("should_clear_canvas", False):
                st.session_state.should_clear_canvas = False
            
            # Create canvas container
           # st.markdown("<div class='canvas-container'>", unsafe_allow_html=True)
            canvas_result = st_canvas(
                fill_color="white",
                stroke_width=20,
                stroke_color="black",
                background_color="white",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key=canvas_key,
            )
            # st.markdown("</div>", unsafe_allow_html=True)
            
            # Buttons row with custom styling
            col_b1, col_b2 = st.columns([1, 1])
            with col_b1:
                if st.button("🗑️ Clear", key="clear_canvas"):
                    st.session_state.should_clear_canvas = True
                    st.rerun()
               
            
            with col_b2:
                predict_btn = st.button("🔍 Predict", key="predict_canvas")
        
        
        with col2:
            if canvas_result.image_data is not None and predict_btn:
                #st.markdown("<div class='result-container'>", unsafe_allow_html=True)
               # with st.spinner('Processing...'):
                    # Convert canvas data to PIL Image
                    image = Image.fromarray(canvas_result.image_data.astype('uint8'))
                    
                    # Process image
                    img_array = preprocess_image(image, invert=True)
                    if img_array is not None:
                        #st.markdown("<div class='preprocessed-image'>", unsafe_allow_html=True)
                        st.image(img_array.reshape(28, 28), width=150)
                        #st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Get predictions
                        result = get_prediction_data(img_array, model)
                        if result:
                            # Display predicted digit
                            st.markdown(f"<div class='prediction-text'>{result['prediction']}</div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='confidence-text'>Confidence: {result['confidence']:.2f}</div>", unsafe_allow_html=True)
                            
                            prob_chart = create_probability_chart(result['probabilities'])
                            if prob_chart:
                                st.pyplot(prob_chart)
                            
                            for idx, pred in enumerate(result['top3']):
                                st.write(f"{idx+1}. Digit {pred['digit']}: {pred['confidence']:.4f}")
                #st.markdown("</div>", unsafe_allow_html=True)
    
    # Tab 2: About
    with tab2:
        st.markdown("<div class='sub-header'>About the Model</div>", unsafe_allow_html=True)
        
        st.write("""
        This application uses a Convolutional Neural Network (CNN) trained on the MNIST dataset
        of handwritten digits. The MNIST dataset is a large collection of handwritten digits that
        is commonly used for training various image processing systems.
        """)
        
        st.markdown("<div class='sub-header'>Model Architecture</div>", unsafe_allow_html=True)
        
        model_summary = '''Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                      │ (None, 26, 26, 32)          │             320 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization                  │ (None, 26, 26, 32)          │             128 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (None, 13, 13, 32)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (None, 11, 11, 64)          │          18,496 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_1                │ (None, 11, 11, 64)          │             256 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 5, 5, 64)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 1600)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 128)                 │         204,928 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 128)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 10)                  │           1,290 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
Total params: 675,872 (2.58 MB)
Trainable params: 225,226 (879.79 KB)
Non-trainable params: 192 (768.00 B)
Optimizer params: 450,454 (1.72 MB)'''
        
        st.code(model_summary, language="")
        
        st.markdown("<div class='sub-header'>Model Code</div>", unsafe_allow_html=True)
        model_code = '''# Build model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])'''
        
        st.code(model_code, language='python')
        
        st.markdown("<div class='sub-header'>Image Preprocessing</div>", unsafe_allow_html=True)
        st.write("""
        When you draw a digit, the application:
        1. Converts it to grayscale
        2. Resizes to 28x28 pixels (MNIST format)
        3. Inverts colors if needed (MNIST expects white digits on black background)
        4. Normalizes pixel values to range [0-1]
        5. Reshapes for CNN input
        """)
        
        st.markdown("<div class='sub-header'>Tech Stack</div>", unsafe_allow_html=True)
        st.write("""
        - **Frontend**: Streamlit for the web interface
        - **Backend**: TensorFlow/Keras for the neural network model
        - **Image Processing**: PIL (Python Imaging Library)
        - **Visualization**: Matplotlib for plotting probabilities
        """)

if __name__ == "__main__":
    main()