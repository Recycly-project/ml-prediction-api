from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
from google.cloud import storage
from io import BytesIO
import tempfile
from functools import lru_cache
import logging


# Konfigurasi aplikasi dan logger
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load konfigurasi dari variabel lingkungan
BUCKET_NAME = os.getenv('BUCKET_NAME', 'model-ml-recycly-bucket')
MODEL_PATH = os.getenv('MODEL_PATH', 'model-prod/recycly-model-2.h5')
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.65))

# Nama kelas yang digunakan dalam prediksi
CLASS_NAMES = ['Botol Utuh 1', 'Botol Rusak', 'Botol Utuh 2', 'Bukan Botol']


def model_exists_in_gcs(bucket_name, model_path):
    try:
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(model_path)
        exists = blob.exists()
        logger.info(f"Model {'found' if exists else 'not found'} at {model_path}")
        return exists
    except DefaultCredentialsError:
        logger.error("Google Cloud credentials not found. Set GOOGLE_APPLICATION_CREDENTIALS.")
        raise RuntimeError("Google Cloud credentials are missing or invalid.")
    except Exception as e:
        logger.error(f"Error checking model in GCS: {e}")
        return False


# Function to check if the model exists in GCS
def model_exists_in_gcs(bucket_name, model_path):
    try:
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(model_path)
        exists = blob.exists()
        logger.info(f"Model {'found' if exists else 'not found'} at {model_path}")
        return exists
    except Exception as e:
        logger.error(f"Error checking model in GCS: {e}")
        return False


# Function to load the model from GCS
@lru_cache(maxsize=1)  # Cache model to avoid repeated loads
def load_model_from_gcs(bucket_name, model_path):
    try:
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(model_path)
        model_bytes = blob.download_as_bytes()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_file:
            temp_file.write(model_bytes)
            temp_file_path = temp_file.name

        logger.info(f"Model loaded from {temp_file_path}")
        return load_model(temp_file_path)
    except Exception as e:
        logger.error(f"Error loading model from GCS: {e}")
        raise RuntimeError("Failed to load model from GCS") from e


# Function to preprocess the image
def preprocess_image(image):
    try:
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        return image.reshape(1, 224, 224, 3)
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise ValueError("Invalid image format") from e


# Validate input image
def validate_image(file):
    try:
        image = Image.open(file.stream)
        if image.format not in ['JPEG', 'PNG']:
            raise ValueError("Unsupported image format. Use JPEG or PNG.")
        return image
    except Exception as e:
        logger.error(f"Error validating image: {e}")
        raise ValueError("Invalid image file") from e


@app.route('/verifyWasteCollection', methods=['POST'])
def verify_waste_collection():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        file = request.files['image']
        image = validate_image(file)
        processed_image = preprocess_image(image)

        if not model_exists_in_gcs(BUCKET_NAME, MODEL_PATH):
            return jsonify({'error': 'Model not found in GCS'}), 500

        model = load_model_from_gcs(BUCKET_NAME, MODEL_PATH)
        prediction = model.predict(processed_image)
        predicted_class_idx = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))

        # Calculate points and status based on prediction
        if CLASS_NAMES[predicted_class_idx] == 'Botol Utuh 1':
            points = 2 if confidence >= CONFIDENCE_THRESHOLD else 0
            status = "botol diterima" if points > 0 else "botol ditolak"
        elif CLASS_NAMES[predicted_class_idx] == 'Botol Utuh 2':
            points = 1
            status = "botol diterima"
        else:
            points = 0
            status = "botol ditolak"

        return jsonify({
            'label': CLASS_NAMES[predicted_class_idx],
            'status': status,
            'confidence': confidence,
            'points': points
        }), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/healthz', methods=['GET'])
def health_check():
    return jsonify({'status': 'OK', 'message': 'Service is running'}), 200



if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host="0.0.0.0", port=port)
