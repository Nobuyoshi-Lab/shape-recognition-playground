from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from constants import SHAPE_NAMES, ADDITIONAL_SHAPE_NAMES
from model import create_model
from keras.models import load_model


app = Flask(__name__)

# Prepare the CNN model
model = create_model()

# Load the pre-trained weights (you need to train the CNN and save the
# weights beforehand)
model = load_model('shape_recognition_model.h5')
shape_names = SHAPE_NAMES + ADDITIONAL_SHAPE_NAMES

@app.route('/')
def index():
    return render_template('index.html', shape_names=shape_names)


@app.route('/recognize_shape', methods=['POST'])
def recognize_shape():
    img_data = base64.b64decode(request.form['image'])
    image = cv2.imdecode(
        np.frombuffer(
            img_data,
            dtype=np.uint8),
        cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize and normalize the image
    gray_resized = cv2.resize(gray, (50, 50))
    gray_normalized = gray_resized / 255.0

    # Add the batch and channel dimensions
    input_image = np.expand_dims(gray_normalized, axis=(0, -1))

    # Predict the shape
    predictions = model.predict(input_image)
    shape_idx = np.argmax(predictions)

    # Get the predicted shape and confidence value
    shape = shape_names[shape_idx]
    confidence = predictions[0][shape_idx]

    return jsonify({'shape': shape, 'confidence': float(confidence)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
