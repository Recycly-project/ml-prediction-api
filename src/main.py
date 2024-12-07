from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
from google.cloud import storage
from io import BytesIO
import tempfile

app = Flask(__name__)


# Function to check if the model exists in GCS
def model_exists_in_gcs(bucket_name, model_path):
    try:
        # Initialize the GCS client
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(model_path)

        # Check if the model exists
        if blob.exists():
            print(f"Model found at {model_path}")
            return True
        else:
            print(f"Model not found at {model_path}")
            return False
    except Exception as e:
        print(f"Error checking model in GCS: {e}")
        return False

# Function to load the model from GCS
def load_model_from_gcs(bucket_name, model_path):
    try:
        # Initialize the GCS client
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(model_path)

        # Download the model file into memory as bytes
        model_bytes = blob.download_as_bytes()

        # Save the model bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_file:
            temp_file.write(model_bytes)
            temp_file_path = temp_file.name
            print(f"Model saved to temporary file: {temp_file_path}")

        # Load the model from the temporary file path
        model = load_model(temp_file_path)
        return model
    except Exception as e:
        print(f"Error loading model from GCS: {e}")
        return None

# Specify your GCS bucket and model path
bucket_name = 'model-ml-recycly-bucket'
model_path = "model-prod/recycly-model-2.h5"

# Check if the model exists in GCS before loading it
if model_exists_in_gcs(bucket_name, model_path):
    try:
        model = load_model_from_gcs(bucket_name, model_path)
        # Print model information (e.g., summary)
        if model:
            print("Model loaded successfully.")
            model.summary()  # This will print the model architecture
            class_names = ['Botol Utuh 1', 'Botol Rusak', 'Botol Utuh 2', 'Bukan Botol']
        else:
            print("Failed to load the model.")
            class_names = []
    except Exception as e:
        model = None
        class_names = []
        print(f"Error loading model: {e}")
else:
    model = None
    class_names = []
    print("Model does not exist in GCS.")


# Preprocess function with error handling
def preprocess_image(image):
    try:
        image = image.resize((224, 224))  # Sesuaikan ukuran input model
        image = np.array(image) / 255.0  # Normalisasi
        return image.reshape(1, 224, 224, 3)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise ValueError("Error preprocessing the image") from e

# Endpoint prediction with error handling
@app.route('/verifyWasteCollection', methods=['POST'])
def verifyWasteCollection():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        file = request.files['image']
        image = Image.open(file.stream)
        processed_image = preprocess_image(image)
        model = load_model_from_gcs(bucket_name, model_path)
        
        # Use model.predict() for prediction
        if model is None:
            return jsonify({'error': 'Model not loaded properly'}), 500
        
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))

        if class_names[predicted_class] == 'Botol Utuh 1':
            points = 2 if confidence >= 0.65 else 0
        elif class_names[predicted_class] == 'Botol Rusak':
            points = 0
        elif class_names[predicted_class] == 'Botol Utuh 2':
            points = 1
        else:
            points = 0

        if class_names[predicted_class] == 'Botol Utuh 1' and confidence < 0.65:
            other_class_idx = np.argmax(np.delete(prediction, 0))
            predicted_class = other_class_idx + 1
            confidence = float(np.max(prediction))

            if class_names[predicted_class] == 'Botol Rusak':
                points = 0
            elif class_names[predicted_class] == 'Botol Utuh 2':
                points = 1
            else:
                points = 0

        return jsonify({
            'label': class_names[predicted_class],
            # 'confidence': confidence,
            'points': points
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))