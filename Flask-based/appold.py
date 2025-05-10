from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import io
import base64
import os

app = Flask(__name__)

# Load once at startup
model = tf.keras.models.load_model('best_mnist_model.h5')

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(img_data, is_file=False):
    if is_file:
        image = Image.open(img_data).convert('L')
    else:
        image = Image.open(io.BytesIO(img_data)).convert('L')
    image = image.resize((28,28), Image.LANCZOS)
    image = ImageOps.invert(image)
    arr = np.array(image).astype('float32') / 255.0
    return arr.reshape(1,28,28,1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data_url = request.json['imageData']
    header, encoded = data_url.split(',', 1)
    img_bytes = base64.b64decode(encoded)
    img_arr = preprocess_image(img_bytes)
    preds = model.predict(img_arr, verbose=0)[0]
    pred_digit = int(np.argmax(preds))
    confidence = float(preds[pred_digit])
    top3_idx = preds.argsort()[-3:][::-1]
    top3 = [{'digit': int(i), 'confidence': float(preds[i])} for i in top3_idx]
    return jsonify({
        'prediction': pred_digit,
        'confidence': confidence,
        'probabilities': preds.tolist(),
        'top3': top3
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        # Save the file temporarily
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Process the image
        img_arr = preprocess_image(filepath, is_file=True)
        preds = model.predict(img_arr, verbose=0)[0]
        pred_digit = int(np.argmax(preds))
        confidence = float(preds[pred_digit])
        top3_idx = preds.argsort()[-3:][::-1]
        top3 = [{'digit': int(i), 'confidence': float(preds[i])} for i in top3_idx]
        
        # Clean up the temporary file
        os.remove(filepath)
        
        return jsonify({
            'prediction': pred_digit,
            'confidence': confidence,
            'probabilities': preds.tolist(),
            'top3': top3
        })

if __name__ == '__main__':
    app.run(debug=True)
