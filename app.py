from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import io
import base64

app = Flask(__name__)

# Load once at startup
model = tf.keras.models.load_model('best_mnist_model.h5')

def preprocess_image(img_bytes):
    # Open the image, convert to grayscale, resize, invert colors, normalize
    image = Image.open(io.BytesIO(img_bytes)).convert('L')
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
    
    # Convert processed image back to base64 for display
    processed_img = Image.fromarray((img_arr[0,:,:,0] * 255).astype('uint8'))
    buffered = io.BytesIO()
    processed_img.save(buffered, format="PNG")
    processed_img_str = base64.b64encode(buffered.getvalue()).decode()
    
    preds = model.predict(img_arr, verbose=0)[0]
    pred_digit = int(np.argmax(preds))
    confidence = float(preds[pred_digit])
    top3_idx = preds.argsort()[-3:][::-1]
    top3 = [{'digit': int(i), 'confidence': float(preds[i])} for i in top3_idx]
    return jsonify({
        'prediction': pred_digit,
        'confidence': confidence,
        'probabilities': preds.tolist(),
        'top3': top3,
        'processedImage': f'data:image/png;base64,{processed_img_str}'
    })

if __name__ == '__main__':
    app.run(debug=True)
