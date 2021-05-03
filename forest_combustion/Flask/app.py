

from __future__ import division, print_function
# coding=utf-8

import os

import numpy as np
from keras.preprocessing import image 
import cv2


from keras.models import load_model
from skimage.transform import resize
from playsound import playsound


import tensorflow as tf

global graph
graph=tf.get_default_graph()

#global graph
#graph = tf.get_default_graph()



# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()

# Load your trained model
model = load_model('forest1.h5')
       # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')




@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('base.html')




@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img = image.load_img(file_path, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        with graph.as_default():
            preds = model.predict_classes(x)
            #preds = [0]
            #type(preds)
            #print(preds[0])
            #print("prediction",preds)
        index = ['Forest With Fire','Forest Without Fire']
        #text = index[0]
        #print(text)
        text = "Prediction : "+index[preds[0]]
        if index[preds[0]] =='Forest With Fire':
            playsound(r'C:\Users\AMAN\Dropbox\My PC (LAPTOP-K3QKQ8QO)\Downloads\https___www.tones7.com_media_tornado_alarm.mp3')
               # ImageNet Decode
    
        
        return text
        
    

from keras.models import load_model
model =load_model('forest1.h5')
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
def detect(frame):
    try:
        img= resize(frame,(64,64))
        img = np.expand_dims(img,axis=0)
        if(np.max(img)>1):
            img =img/255.0
        prediction =model.predict(img)
        print (prediction)
        prediction_class = model.predict_classes(img)
        print(prediction_class)
        return prediction_class
    except AttributeError:
        print("shape not found")  

  


if __name__ == '__main__':
    app.run(debug = False,threaded = False)
    
#frame= cv2.imread(r'C:\Users\AMAN\Dropbox\My PC (LAPTOP-K3QKQ8QO)\Downloads\data\data\train\Fire\fire-45.7396875153.png')


# from playsound import playsound
#if(data[0]==0):
    #playsound(r'C:\Users\AMAN\Dropbox\My PC (LAPTOP-K3QKQ8QO)\Downloads\https___www.tones7.com_media_tornado_alarm.mp3')
#elif(data[0]==1):
    #playsound(r'C:\Users\AMAN\Dropbox\My PC (LAPTOP-K3QKQ8QO)\Downloads\https___www.tones7.com_media_tornado_alarm.mp3')




