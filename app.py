from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import joblib
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model and scaler
model = joblib.load('svm_model.pkl')

TARGET_SIZE = (128, 128)

# Function to preprocess the image (convert to grayscale)
def preprocess_image(image_path, target_size=(128, 128)):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize(target_size)  # Resize to match the training images
    img_array = np.array(img).flatten()  # Flatten the image
    return img_array.reshape(1, -1)  # Reshape to a 2D array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        file_path = os.path.join('/tmp', file.filename)
        file.save(file_path)

        # Preprocess the image
        img_array = preprocess_image(file_path)

        # Predict using the SVM model
        prediction = model.predict(img_array)
        predicted_label = 'normal' if prediction[0] == 0 else 'disorder'
        print(predicted_label)


        return jsonify({'prediction': predicted_label})

    return jsonify({'error': 'File not found'}), 400

if __name__ == '__main__':
    app.run(debug=True)
