from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Muat model
model = tf.keras.models.load_model('recycly_model.h5')
class_names = ['Botol Diterima', 'Botol Rusak', 'Bukan Botol']  # Sesuaikan dengan training

# Fungsi preprocess
def preprocess_image(image):
    image = image.resize((128, 128))  # Sesuaikan ukuran input model
    image = np.array(image) / 255.0  # Normalisasi
    return image.reshape(1, 128, 128, 3)

# Endpoint prediksi
@app.route('/verifyWasteCollection', methods=['POST'])
def verifyWasteCollection():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    image = Image.open(file.stream)
    processed_image = preprocess_image(image)

    prediction = model.verifyWasteCollection(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    # confidence = float(np.max(prediction))

    # Logika pemberian poin
    points = 10 if predicted_class == 0 else 0

    return jsonify({
        'label': class_names[predicted_class],
        # 'confidence': confidence,
        'points': points
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)