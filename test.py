import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
import numpy as np
import io

app = Flask(__name__)

model_path = r'C:\Users\ahmee\OneDrive\Masaüstü\YL\Pressure_Injury\kod\saved_models/DenseNet121_pressure_injury_classifier.keras'
model = tf.keras.models.load_model(model_path)

classes = ['Evre 1', 'Evre 2', 'Evre 3', 'Evre 4']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        img_bytes = io.BytesIO(file.read())
        
        img = image.load_img(img_bytes, target_size=(150, 150))
        
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)
        
        predicted_class_index = np.argmax(predictions[0])
        
        predicted_class = classes[predicted_class_index]
        
        response = {
            'prediction': predicted_class,
            'probabilities': {classes[i]: float(predictions[0][i]) for i in range(len(classes))}
        }
        
        return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
