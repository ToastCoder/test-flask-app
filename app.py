# IMPORTING REQUIRED LIBRARIES
from flask import Flask, render_template, request
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2
import numpy as np
from flask_cors import CORS, cross_origin
from PIL import Image
import tensorflow as tf
import os

classes = ["Giloma", "Meningiloma", "Normal", "Pitutary"]

# FUNCTION TO PREDICT CLASS
def predict(IMG_PATH):
    model = load_model("tumor-model")
    
    data_dir = os.path.join(os.getcwd(),"samples/")
    images = []
    print(data_dir)
    for img in os.listdir(data_dir):
        img = os.path.join(data_dir,img)
        img = tf.keras.preprocessing.image.load_img(data_dir, target_size=(200, 200), grayscale = True)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        images.append(img)
    files = os.listdir(os.getcwd())
    images = np.vstack(images)

    res = model.predict(images)
    label = np.argmax(res)
    result = classes[label]
    #result = data_dir
    return result

# INITIALIZING FLASK APP
app = Flask(__name__)
cors = CORS(app)

@app.route("/")
def main():
    return "Application running."

@app.route("/process",methods = ["POST"])
def processFn():
    data = request.files["img"]
    data.save("samples/img.jpg")
    res = predict("samples/img.ipg")
    return res

if __name__ == "__main__":
    app.run(debug = True)

