from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from constants import (SHAPE_NAMES, ADDITIONAL_SHAPE_NAMES, INPUT_SHAPE)
from model import create_model
from keras.models import load_model

app = Flask(__name__)
CORS(app)

model = create_model()
model = load_model('shape_recognition_model.h5')
shape_names = SHAPE_NAMES + ADDITIONAL_SHAPE_NAMES

@app.route('/')
def index():
    return render_template('index.html', shape_names=shape_names)

@app.route('/recognize_shape', methods=['POST'])
def recognize_shape():
    img_data = base64.b64decode(request.form['image'])
    image = cv2.imdecode(np.frombuffer(img_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    cv2.imwrite("input_image_debug.png", image)  # Save input image

    # Pre-processing the input image
    bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    alpha = image[:, :, 3]

    # Create a binary image using alpha channel
    _, binary = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)
    binary_white_bg = cv2.bitwise_not(binary)
    gray_shape = cv2.bitwise_and(gray, binary)
    preprocessed_image = cv2.add(binary_white_bg, gray_shape)

    cv2.imwrite("preprocessed_image_debug.png", preprocessed_image)  # Save preprocessed image

    # Resizing the input image with cv2.INTER_AREA interpolation method
    gray_resized = cv2.resize(preprocessed_image, INPUT_SHAPE[:2], interpolation=cv2.INTER_AREA)

    cv2.imwrite("gray_resized_debug.png", gray_resized)  # Save resized grayscale image

    # Predicting the shape
    gray_normalized = gray_resized / 255.0
    gray_normalized = np.expand_dims(gray_normalized, axis=-1)
    gray_normalized = np.expand_dims(gray_normalized, axis=0)

    predictions = model.predict(gray_normalized)
    shape_idx = np.argmax(predictions)
    shape = shape_names[shape_idx]
    confidence = predictions[0][shape_idx]

    # Post-processing the output
    confidence_threshold = 0.75
    if confidence < confidence_threshold:
        shape = "unknown"

    return jsonify({'shape': shape, 'confidence': float(confidence)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
