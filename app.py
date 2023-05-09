from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from constants import SHAPE_NAMES, ADDITIONAL_SHAPE_NAMES
from model import create_model
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator


def augment_image(image, datagen):
    image_batch = np.expand_dims(image, axis=0)
    augmented_image = next(datagen.flow(image_batch, batch_size=1))[0]
    return augmented_image


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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (50, 50))
    gray_normalized = gray_resized / 255.0
    gray_normalized = np.expand_dims(gray_normalized, axis=-1)
    gray_normalized = np.expand_dims(gray_normalized, axis=0)

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1)
    augmented_image = augment_image(gray_normalized[0], datagen)

    predictions = model.predict(np.expand_dims(augmented_image, axis=0))
    shape_idx = np.argmax(predictions)
    shape = shape_names[shape_idx]
    confidence = predictions[0][shape_idx]

    return jsonify({'shape': shape, 'confidence': float(confidence)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
