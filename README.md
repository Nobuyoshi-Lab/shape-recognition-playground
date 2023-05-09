# Shape Recognition Playground

This project is a web application for recognizing hand-drawn shapes using a deep learning model. The application is built using Flask, and the model is trained using Keras.

## Credits

This project uses the following resources:

- [Flask](https://flask.palletsprojects.com/)
- [Tensorflow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [OpenCV](https://opencv.org/)
- [Quick, Draw! Dataset](https://quickdraw.readthedocs.io/)

## Prerequisites

1. Clone the repository.

```
git clone https://github.com/Nobuyoshi-Lab/shape-recognition-playground.git
cd shape_recognition_playground
```

## Train the Model

There is already a trained model in the `shapes_dataset` folder. If you want to train the model yourself, you can do so by following the steps below.

To train the model, follow these steps:

1. Install the additional required packages for training.

```
pip install -r shapes_dataset/requirements.txt
```

2. Run the `weight_training.py` script in the `shapes_dataset` folder.

```
python shapes_dataset/weight_training.py
```

3. (Optional) Run the `generate_constants.py` script in the `shapes_dataset` folder to generate the random labels for the dataset and save them to the `constants.py` file.

```
python shapes_dataset/generate_constants.py
```

You can also change the `ADDITIONAL_SHAPE_AMOUNT` in the `constants.py` script in the `app` folder to change the number of additional shapes to generate.

## Run the application.

### Local

```
python app/app.py
```

### Docker

Windows:

```
docker-compose up --build
```

Linux:

```
sudo docker compose up --build
```

Now you can access the web application at `http://localhost:5000`.

## Usage

Draw a shape on the canvas and click "Recognize Shape". The application will recognize the shape and it might play a sound corresponding to the recognized shape.
