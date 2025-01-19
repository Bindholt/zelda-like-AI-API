import tensorflow as tf
import numpy as np
from PIL import Image
import mss
import mss.tools
from io import BytesIO
from flask import Flask, jsonify
from flask_cors import CORS
import time
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from pynput import keyboard
from pynput import mouse
import threading

# Create and configure app before anything else
app = Flask(__name__)
CORS(app)

# Load your Keras model
model = tf.keras.models.load_model('model.h5')

# Define IMG_SIZE based on the model's expected input size
IMG_SIZE = [300, 300]  # Adjust this to your model's input size

# Global variable for region
region = {"top": 0, "left": 0, "width": 0, "height": 0}

def define_region():
    """Capture two mouse clicks to define the screenshot region."""
    coordinates = []

    def on_click(x, y, button, pressed):
        if pressed:
            print(f"Mouse clicked at ({x}, {y})")
            coordinates.append((x, y))
            if len(coordinates) == 2:
                # Calculate region from the two points
                x1, y1 = coordinates[0]
                x2, y2 = coordinates[1]
                region["top"] = min(y1, y2)
                region["left"] = min(x1, x2)
                region["width"] = abs(x2 - x1)
                region["height"] = abs(y2 - y1)
                print(f"Region defined: {region}")
                return False  # Stop listening after 2 clicks

    with mouse.Listener(on_click=on_click) as listener:
        listener.join()

def capture_screenshot():
    """Capture a screenshot of the specified region and return it as an in-memory image."""
    with mss.mss() as sct:
        # Capture the screen
        screenshot = sct.grab(region)
        # Convert the screenshot to a PIL Image
        mss.tools.to_png(screenshot.rgb, screenshot.size, output='gamestate.png')

def on_press(key):
    """Handle key press events."""
    try:
        if key.char == 'n':  # Check if 'n' was pressed
            print("Press 'n' detected. Defining region...")
            define_region()
    except AttributeError:
        pass

def start_key_listener():
    """Start the keyboard listener in a non-blocking way."""
    listener = keyboard.Listener(on_press=on_press)
    listener.start()  # This starts the listener in a separate thread
    return listener

@app.route('/predict', methods=['GET'])
def predict():
    """Endpoint for making predictions based on a screenshot."""
    img = image.load_img('gamestate.png', target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    print(f"Predicted move: {predictions}")
    input_mapping = ['w', 'a', 's', 'd', 'e']

    predicted_input = input_mapping[np.argmax(predictions)]

    return jsonify({'input': predicted_input})


print("Press 'n' to define the screenshot region.")
key_listener = start_key_listener()


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)