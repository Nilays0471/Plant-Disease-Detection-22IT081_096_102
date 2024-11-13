import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
#from tensorflow.keras.models import load_model
#from tensorflow import keras
#from keras._tf_keras.keras.preprossing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

load_model = tf.keras.models.load_model

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'Wheat_Disease_Detection.keras'
model = load_model(MODEL_PATH)

# Endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    # Save the file
    filename = secure_filename(file.filename)
    file_path = os.path.join('uploads', filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(file_path)
    
    # Preprocess the image
    img = image.load_img(MODEL_PATH, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    
    # Clean up the saved file
    os.remove(file_path)
    
    # Respond with the prediction
    return jsonify({"predicted_class": int(predicted_class[0])})

if __name__ == '__main__':
    app.run(debug=True)
