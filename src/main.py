from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Muat model dengan error handling
try:
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'recycly_model.h5')
    model = load_model(model_path)
    class_names = ['Botol Diterima', 'Botol Rusak', 'Bukan Botol']  # Sesuaikan dengan training
except Exception as e:
    model = None
    class_names = []
    print(f"Error loading model: {e}")

# Fungsi preprocess dengan error handling
def preprocess_image(image):
    try:
        image = image.resize((128, 128))  # Sesuaikan ukuran input model
        image = np.array(image) / 255.0  # Normalisasi
        return image.reshape(1, 128, 128, 3)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise ValueError("Error preprocessing the image") from e

# Endpoint prediksi dengan error handling
@app.route('/verifyWasteCollection', methods=['POST'])
def verifyWasteCollection():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        file = request.files['image']
        image = Image.open(file.stream)
        processed_image = preprocess_image(image)
        
        # Gunakan model.predict() untuk prediksi
        if model is None:
            return jsonify({'error': 'Model not loaded properly'}), 500
        
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))

        # Logika pemberian poin
        points = 10 if predicted_class == 0 else 0

        return jsonify({
            'label': class_names[predicted_class],
            'confidence': confidence,
            'points': points
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
