import tensorflow as tf
import numpy as np
from PIL import Image
import mss
import mss.tools
from io import BytesIO
from flask import Flask, jsonify
import time
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input


app = Flask(__name__)

# Load your Keras model
model = tf.keras.models.load_model('model.h5')

# Define IMG_SIZE based on the model's expected input size
IMG_SIZE = [300, 300] # Adjust this to your model's input size

# Capture screenshot logic (using mss)
def capture_screenshot():
    """Capture a screenshot of the specified region and return it as an in-memory image."""
    with mss.mss() as sct:
        # Define the region to capture
        region = {"top": 95, "left": 0, "width": 770, "height": 315}

        # Capture the screen
        screenshot = sct.grab(region)

        # Convert the screenshot to a PIL Image
        mss.tools.to_png(screenshot.rgb, screenshot.size, output='gamestate.png')

@app.route('/predict', methods=['GET'])
def predict():
    # Capture screenshot of the game window
    capture_screenshot()

    img = image.load_img('gamestate.png', target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    print(f"Predicted move: {predictions}")
    input_mapping = ['w', 'a', 's', 'd', 'e']

    predicted_input = input_mapping[np.argmax(predictions)]

    return jsonify({'input': predicted_input})


if __name__ == '__main__':
    app.run(debug=True)



