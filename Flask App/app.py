import os
import sys
import cv2
# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil


# Declare a flask app
app = Flask(__name__)


# You can use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.mobilenet_v2 import MobileNetV2
#model = MobileNetV2(weights='imagenet')



# Model saved with Keras model.save()
MODEL_PATH1 = 'models/Resnetmodel(covvsnorvsothers)'
MODEL_PATH2 = 'models/2classResnetBacvsViru.hdf5'
MODEL_PATH3 = 'models/Resnetmodel(normalvspneumonia)'


# Load your own trained model
model1 = load_model(MODEL_PATH1)
model2 = load_model(MODEL_PATH2)
model3 = load_model(MODEL_PATH3)
print(model1)

#model_predict(img,model1) # Necessary
#model2._make_predict_function()
#model3._make_predict_function()
print('Model loaded. Start serving...Check http://127.0.0.1:5000/')


def model_predict(img, model):
    
    img = img.resize((224, 224))
    img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2RGB) # arrange format as per keras
    # Preprocessing the image
    x = image.img_to_array(img)
    
    #x = np.array(img) / 255
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='tf')

    preds = model.predict(x)
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
        preds1 = model_predict(img, model1)
        preds2 = model_predict(img, model2)
        preds3 = model_predict(img, model3)
        print('Predicted Probabilities of Model 1')
        print(preds1)
        print('Predicted Probabilities of Model 2')
        print(preds3)
        print('Predicted Probabilities of Model 3')
        print(preds2)
        pred_class1 = np.argmax(preds1)
        pred_class2 = np.argmax(preds2)
        pred_class3 = np.argmax(preds3)
        # Process your result for human
        pred_proba1 = "{:.3f}".format(np.amax(preds1))    # Max probability
        pred_proba2 = "{:.3f}".format(np.amax(preds2))
        pred_proba3 = "{:.3f}".format(np.amax(preds3))
        if pred_class1 == 0:
            result = 'You are COVID-19 positive!!!'
            print('Predictedclass: ')
            print('Covid')
        elif pred_class1 == 1:
            if pred_class3 == 0:    
                 result = 'NO INFECTION DETECTED!!!'
                 print('Predictedclass: ')
                 print('Normal')
            if pred_class3 == 1:
                 if pred_class2 == 0:
                    result='You have PNEUMONIA!!! \n\nCategory: Bacteria \n\n(Bacterial probability: '+pred_proba2+')'
                    print('Predictedclass: ')
                    print('Bacterial Pneumonia')
                 elif pred_class2 == 1:
                     result='You have PNEUMONIA!!! \n\nCategory: Virus \n\n(Viral probability: '+pred_proba2+')'
                     print('Predictedclass: ')
                     print('Viral Pneumonia')
        elif pred_class1 == 2:
             if pred_class3 == 0:    
                 result = 'NO INFECTION DETECTED!!!'
                 print('Predictedclass: ')
                 print('Normal')
             if pred_class3 == 1:
                  if pred_class2 == 0:
                     result='You have PNEUMONIA!!! \n\nCategory: Bacteria \n\n(Bacterial probability: '+pred_proba2+')'
                     print('Predictedclass: ')
                     print('Bacterial Pneumonia')
                  elif pred_class2 == 1:
                     result='You have PNEUMONIA!!! \n\nCategory: Virus \n\n(Viral probability: '+pred_proba2+')'
                     print('Predictedclass: ')
                     print('Viral Pneumonia')
        #result = result.replace('_', ' ').capitalize()
        
        # Serialize the result, you can add additional fields
        if pred_class1 == 2 or pred_class1 == 1:
             if pred_class3 == 0:
                 return jsonify(result=result, probability=pred_proba1)
             if pred_class3 == 1:   
                 return jsonify(result=result, probability=pred_proba2)
        else:
           return jsonify(result=result, probability=pred_proba1)
    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
