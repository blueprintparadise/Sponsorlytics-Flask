import base64
import json
import os
import sys
import sys
#sys.path.insert(0, './yolov5')
from io import BytesIO

import matplotlib.pyplot as plt
import torch
# Model
from PIL import Image

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
# model = torch.load(r'yolov5s.pt')
# model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'D:\Sponsorlytix\Sponsorlytix_Flask_Team\best.pt', force_reload=True)

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil


# Declare a flask app
app = Flask(__name__)


# You can use pretrained model from Keras
# Check https://keras.io/applications/
# or https://www.tensorflow.org/api_docs/python/tf/keras/applications

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
#model = MobileNetV2(weights='imagenet')

print('Model loaded. Check http://127.0.0.1:5000/')


# Model saved with Keras model.save()
#MODEL_PATH = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load your own trained model
#model = torch.load(MODEL_PATH)
#model(img)          # Necessary
print('Model loaded. Start serving...')


def model_predict(img, model):
    img = img.resize((224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)

    im = Image.fromarray((x * 255).astype(np.uint8))

    preds = model(im)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # Make prediction
        preds = model_predict(img, model)

        # Process your result for human
        preds.render()  # updates results.imgs with boxes and labels
        crops = preds.crop(save=True)
        report = preds.pandas().xyxy[0].to_json(orient="records")

        return jsonify(result=str(report))

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
