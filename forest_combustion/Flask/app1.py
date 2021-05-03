from __future__ import division, print_function
from flask import Flask, request, render_template, jsonify
#from werkzeug import secure_filename
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

global model,graph
import tensorflow as tf
import smtplib
import cv2
graph = tf.get_default_graph()
app = Flask(__name__)

#MODEL_PATH = 'cnn_catdog.h5'

#model = load_model('cnn_catdog.h5')

@app.route('/', methods=['GET'])
def index():
    return render_template('base.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    model = load_model('forest1.h5')
    video = cv2.VideoCapture(0)
    name = ['Forest','Forestfire']
    c=0
    while(c==0):
        success, frame = video.read()
        cv2.imwrite(r"C:\Users\AMAN\Dropbox\My PC (LAPTOP-K3QKQ8QO)\Downloads\data\data\train\Fire\fire-45.7396875153.png",frame)
        img = image.load_img(r"C:\Users\AMAN\Dropbox\My PC (LAPTOP-K3QKQ8QO)\Downloads\data\data\train\Fire\fire-45.7396875153.png",target_size = (64,64))
        x  = image.img_to_array(img)
        x = np.expand_dims(x,axis = 0)
        pred = model.predict_classes(x)
        p = pred[0]
        print(p[0])
        c=p[0]
        cv2.putText(frame, "predicted  class = "+str(name[p[0]]), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1)
        cv2.imshow("f_stream",frame)
        if cv2.waitKey(1) & 0xFF == ord('a'): 
            break
    result="Forestfire"
    gmail_user = 'aman.1nh18cs710.cse'   #email id without @gmail.com
    gmail_password = 'babuamanbbs'
    #email properties
    sent_from = gmail_user
    to = ['aman.1nh18cs710.cse@gmail.com']
    email_text = """Alert! forest fire detected.
                    Immediate action required."""
    server_ssl = smtplib.SMTP_SSL('smtp.gmail.com',465)
    server_ssl.ehlo()
    server_ssl.login(gmail_user, gmail_password)
    server_ssl.sendmail(sent_from, to, email_text)
    server_ssl.close()
    print ('Email sent!')
    print(result)
    return render_template('base.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run()