import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
import numpy as np
import io
import os

app = Flask(__name__)

# Path to your saved model
model_path = os.path.join(os.path.dirname(__file__), 'saved_models/best_model.h5')

# Load the model
model = None
try:
    if os.path.isfile(model_path):
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
    else:
        print(f"Model file not found at {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")

# Define the classes
classes = ['Evre 1', 'Evre 2', 'Evre 3', 'Evre 4']

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            img_bytes = io.BytesIO(file.read())
            img = image.load_img(img_bytes, target_size=(150, 150))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make a prediction
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions[0])
            predicted_class = classes[predicted_class_index]
            
            response = {
                'prediction': predicted_class,
                'probabilities': {classes[i]: float(predictions[0][i]) for i in range(len(classes))}
            }
            
            return jsonify(response)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
