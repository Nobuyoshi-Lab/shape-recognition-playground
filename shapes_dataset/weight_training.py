import numpy as np
import cv2
import os
import sys
from quickdraw import QuickDrawData
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from model import create_model
from constants import (
    SHAPE_NAMES,
    ADDITIONAL_SHAPE_NAMES,
    MAX_DRAWINGS,
    TEST_SIZE,
    RANDOM_STATE,
    TRAINING_EPOCHS,
    BATCH_SIZE)


def load_dataset(shape_names):
    qdraw = QuickDrawData(
        recognized=True,
        max_drawings=MAX_DRAWINGS,
        cache_dir="cache")
    images = []
    labels = []

    for shape_idx, shape_name in enumerate(shape_names):
        qdraw_group = qdraw.get_drawing_group(shape_name)
        for drawing in qdraw_group.drawings:
            pil_img = drawing.get_image()
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2GRAY)
            img_resized = cv2.resize(img, (50, 50))
            images.append(img_resized)
            labels.append(shape_idx)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


def preprocess_data(images, labels):
    images = images / 255.0
    images = np.expand_dims(images, axis=-1)
    labels = to_categorical(labels)

    return images, labels


def main():
    shape_names = SHAPE_NAMES + ADDITIONAL_SHAPE_NAMES
    images, labels = load_dataset(shape_names)
    images, labels = preprocess_data(images, labels)
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    model = create_model()

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1)
    datagen.fit(X_train)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True)

    training_epochs = TRAINING_EPOCHS
    model.fit(
        datagen.flow(
            X_train,
            y_train,
            batch_size=BATCH_SIZE),
        epochs=training_epochs,
        validation_data=(
            X_test,
            y_test),
        callbacks=[early_stopping])

    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    model_path = os.path.join(parent_dir, 'shape_recognition_model.h5')
    model.save(model_path)


if __name__ == "__main__":
    main()
