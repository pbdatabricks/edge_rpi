import base64
import io
import tarfile

import requests
from flask import Flask, request, jsonify
import torch
import mlflow
import numpy
from torchvision import datasets, models, transforms
from PIL import Image, ImageDraw
from picamera import PiCamera
from time import sleep
import datetime as dt
import sys
import subprocess
import os
import json
from torch.hub import load_state_dict_from_url
from torchvision.models import mobilenet_v2 as mobilenetv2

app = Flask(__name__)
@app.route('/', methods=['GET'])
def alive():
    timestamp = dt.datetime.now()
    return jsonify({"response":"API is alive", "status":"200", "url":"frostydew1905.cotunnel.com", "timestamp": timestamp})

@app.route('/api/send_url/', methods=['GET', 'POST'])
def get_url():
    content = request.json
    model = content["model"]
    version = content["version"]
    url = content["url"]
    status = "SUCCESS"
    response = requests.get(url, stream=True)
    target_path = '/home/pi/compressed/model.tar.gz'
    timestamp = dt.datetime.now()
    if response.status_code == 200:
        with open(target_path, 'wb') as f:
            f.write(response.raw.read())
    return jsonify({"url": url, "model": model, "version": version, "status": status, "timestamp": timestamp})

@app.route('/api/artifacts/', methods=['GET', 'POST'])
def download_artifacts():
    content = request.json
    model = content["model"]
    version = content["version"]
    data = content["data"]
    string_to_b64 = data.encode('utf-8')
    b64_to_bytes = base64.b64decode(string_to_b64)
    file_like_object = io.BytesIO(b64_to_bytes)
    tar = tarfile.open(fileobj=file_like_object, mode='r:gz')
    dest_path = '/home/pi/artifacts/' + model + '_' + version + '/'
    tar.extractall(path=dest_path)
    timestamp = dt.datetime.now()
    return jsonify({"model": model, "version": version, "status": "production", "dest_path": dest_path, "timestamp": timestamp})

@app.route('/api/inference/', methods=['GET'])
def infer():
    model_path = '/home/pi/artifacts/MobileNetV3_100/MobileNetV3/data/model.pth'
    labels = json.dumps({"0": "empty", "1": "box"})
    CURRENT_DATE = dt.datetime.now().strftime('%m-%d-%Y_%H:%M:%S')
    camera = PiCamera()
    camera.resolution = (600, 600)
    camera.framerate = 15
    camera.start_preview()
    sleep(10)

    image_filepath = '/home/pi/images/' + CURRENT_DATE + '.jpg'
    camera.capture(image_filepath)
    camera.stop_preview()
    camera.close()
    test_image = image_filepath
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    labels = json.loads(labels)
    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(test_image)
    image = data_transform(img).unsqueeze(0).to(device)
    model = torch.load(model_path, pickle_module=mlflow.pytorch.pickle_module, map_location=torch.device('cpu'))
    model.to(device)
    model.eval()
    out = model(image)
    # print(out)
    print(out.argmax().item())
    print("Predicted class is: {}".format(labels[str(out.argmax().item())]))
    prediction = labels[str(out.argmax().item())]
    timestamp = dt.datetime.now()
    d1 = ImageDraw.Draw(img)
    text = "Prediction: " + prediction
    d1.text((28, 36), text, fill=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, 'jpeg')
    buf.seek(0)
    img_bytes = buf.read()
    buf.close()
    return jsonify({"predicted": prediction, "timestamp": timestamp, "image": base64.b64encode(img_bytes).decode('utf-8')})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8080', debug=True)
