import os
import logging
from io import BytesIO

import boto3
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image


#  CONFIGURATION

S3_ENDPOINT = os.getenv('S3_ENDPOINT', 'https://is3.cloudhost.id/')
S3_BUCKET   = os.getenv('S3_BUCKET', 'ml-model-bucket')
MODEL_KEY   = os.getenv('MODEL_KEY', 'recycle_model2.h5')
CONF_THRESH = float(os.getenv('CONF_CONFIDENCE_THRESHOLD', '0.65'))

CLASS_NAMES = [
    'Bottle without bottle cap and label',
    'Bottle Damage',
    'Full Bottle',
    'Non Bottle'
]

# Ambil kredensial dari env vars (pastikan sudah di-set di deployment)
S3_ACCESS_KEY = os.getenv('S3_ACCESS_KEY', '66013J4YPRY7ZDZQN0MC')
S3_SECRET_KEY = os.getenv('S3_SECRET_KEY', 'nGAct7fpIewfBybbbv9XsLxWGSM4u9T36VDN1Vxy')
if not (S3_ACCESS_KEY and S3_SECRET_KEY):
    raise RuntimeError("S3_ACCESS_KEY and S3_SECRET_KEY must be set in environment")


#  LOGGER SETUP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("waste-verifier")


#  S3 & MODEL LOADING

def download_model_from_s3() -> tf.keras.Model:
    logger.info(f"Downloading model from S3-compatible storage: {S3_ENDPOINT}/{S3_BUCKET}/{MODEL_KEY}")

    # Inisialisasi client S3 dengan endpoint kustom
    s3 = boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY
    )

    # Unduh file ke file sementara
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_file:
        s3.download_fileobj(S3_BUCKET, MODEL_KEY, temp_file)
        temp_model_path = temp_file.name

    logger.info(f"Model downloaded successfully to: {temp_model_path}")
    return tf.keras.models.load_model(temp_model_path)

# Load model sekali di startup
MODEL = download_model_from_s3()


#  IMAGE PROCESSING

def preprocess_image(image):
    try:
        # Ubah ukuran gambar ke 224x224 (sesuai dengan yang diinginkan)
        image = image.resize((224, 224))  # Mengubah ukuran gambar ke 224x224
        image = np.array(image) / 255.0   # Normalisasi piksel gambar
        return image.reshape(1, 224, 224, 3)  # Menyesuaikan dengan input model yang diharapkan
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


#  FLASK APP FACTORY

def create_app():
    app = Flask(__name__)

    @app.errorhandler(ValueError)
    def handle_bad_request(e):
        return jsonify({'error': str(e)}), 400

    @app.errorhandler(Exception)
    def handle_internal_error(e):
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

    @app.route('/verifyWasteCollection', methods=['POST'])
    def verify_waste_collection():
        if 'image' not in request.files:
            raise ValueError("No image uploaded under field 'image'")

        img = validate_image(request.files['image'])
        x   = preprocess_image(img)

        preds = MODEL.predict(x)
        idx   = int(np.argmax(preds, axis=1)[0])
        conf  = float(np.max(preds))

        label = CLASS_NAMES[idx]
        # Business logic
        if label == 'Bottle without bottle cap and label':
            points = 2 if conf >= CONF_THRESH else 0
            status = 'Bottle Accepted' if points > 0 else 'Bottle Decline'
        elif label == 'Full Bottle':
            points = 1
            status = 'Bottle Accepted'
        else:
            points = 0
            status = 'Bottle Decline'

        return jsonify({
            'label':      label,
            'status':     status,
            'confidence': round(conf, 4),
            'points':     points
        })

    @app.route('/healthz', methods=['GET'])
    def healthz():
        return jsonify({'status': 'ok'}), 200

    return app


#  ENTRYPOINT

if __name__ == '__main__':
    flask_app = create_app()
    port = int(os.getenv('PORT', 5001))
    flask_app.run(host='0.0.0.0', port=port, debug=False)
