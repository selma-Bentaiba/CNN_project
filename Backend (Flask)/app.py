from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import io
from io import StringIO
import base64
from PIL import Image
import numpy as np
import os
import time
import torch
import torch.nn as nn
from torchvision.transforms import functional as f

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# Gesture classes
targets = {
    1: "call",
    2: "dislike",
    3: "fist",
    4: "four",
    5: "like",
    6: "mute",
    7: "ok",
    8: "one",
    9: "palm",
    10: "peace",
    11: "rock",
    12: "stop",
    13: "stop inverted",
    14: "three",
    15: "two up",
    16: "two up inverted",
    17: "three2",
    18: "peace inverted",
    19: "no gesture"
}

# Simplified model loading function
def load_model(model_path, device="cpu"):
    """
    Load the SSD MobileNet model
    """
    from ssd_mobilenetv3 import SSDMobilenet
    
    ssd_mobilenet = SSDMobilenet(num_classes=len(targets) + 1)
    
    if not os.path.exists(model_path):
        print(f"Model not found {model_path}")
        raise FileNotFoundError

    # Load model weights
    ssd_mobilenet.load_state_dict(torch.load(model_path, map_location=device))
    ssd_mobilenet.eval()
    return ssd_mobilenet

# Image preprocessing function
def preprocess(img):
    """
    Preprocess image for model input
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img)
    width, height = image.size

    image = ImageOps.pad(image, (max(width, height), max(width, height)))
    padded_width, padded_height = image.size
    image = image.resize((320, 320))

    img_tensor = f.pil_to_tensor(image)
    img_tensor = f.convert_image_dtype(img_tensor)
    img_tensor = img_tensor[None, :, :, :]
    return img_tensor, (width, height), (padded_width, padded_height)

# Gesture detection function
def detect_gesture(detector, frame, num_hands=2, threshold=0.5):
    """
    Detect hand gestures in the frame
    """
    # Convert OpenCV BGR format to RGB for PIL
    processed_frame, size, padded_size = preprocess(frame)
    
    with torch.no_grad():
        output = detector(processed_frame)[0]
    
    boxes = output["boxes"][:num_hands]
    scores = output["scores"][:num_hands]
    labels = output["labels"][:num_hands]
    
    # Process the results
    result = ["NO GESTURES"]
    
    for i in range(min(num_hands, len(boxes))):
        if scores[i] > threshold:
            result = [targets[int(labels[i])], float(scores[i])]
            break
    
    return result

# Load the model only once when the app starts
try:
    from PIL import ImageOps
    model = load_model(os.path.expanduser("SSDLite.pth"))
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route("/")
def home():
    return "Gesture Recognition API"

@app.route("/http-call")
def http_call():
    data = {'data': 'This text was fetched using an HTTP call to server on render'}
    return jsonify(data)

@socketio.on("connect")
def connected():
    print(f"Client connected: {request.sid}")
    emit("connect", {"data": f"id: {request.sid} is connected"})

@socketio.on('data')
def handle_message(data):
    print(f"Data from the front end: {str(data)}")
    emit("data", {'data': data, 'id': request.sid}, broadcast=True)

@socketio.on("disconnect")
def disconnected():
    print(f"User disconnected: {request.sid}")
    emit("disconnect", f"user {request.sid} disconnected", broadcast=True)

@socketio.on('image')
def image(data_image):
    if model is None:
        emit('response_back', "Model not loaded")
        return
    
    start = time.time()
    
    # Decode base64 image
    try:
        b = io.BytesIO(base64.b64decode(data_image))
        pimg = Image.open(b)
        frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error decoding image: {e}")
        emit('response_back', "Error processing image")
        return
    
    # Run gesture detection
    try:
        result = detect_gesture(model, frame, num_hands=1, threshold=0.75)
        print(f"Detected gesture: {result[0]}")
    except Exception as e:
        print(f"Error detecting gesture: {e}")
        result = ["Error", str(e)]
    
    end = time.time()
    print(f"Processing time: {end - start:.4f} seconds")
    
    # Emit the detected gesture
    emit('response_back', result[0])

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5001)