# -*- coding: utf-8 -*-
import numpy as np
from io import BytesIO
import time
from flask import Flask, render_template
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
#import tensorflow as tf
#tf.python.control_flow_ops = tf
import json
import argparse
import base64
from datetime import datetime
import os
import shutil
import cv2
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from PIL import ImageOps
from io import BytesIO
from keras.models import model_from_json
from keras.models import load_model
import h5py
from keras import __version__ as keras_version

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

Rows, Cols = 64, 64
num = 0
#steering = []


def smoothing(angles, pre_frame):
    # collect frames & print average line
    angles = np.squeeze(angles)
    avg_angle = 0

    for ii, ang in enumerate(reversed(angles)):
        if ii == pre_frame:
            break
        avg_angle += ang
    avg_angle = avg_angle / pre_frame

    return avg_angle
    
def crop_img(image):
    #crop unnecessary parts 
    cropped_img = image[63:136, 0:319]
    resized_img = cv2.resize(cropped_img, (Cols, Rows))

    img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    return img

class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral





controller = SimplePIController(0.1, 0.002)
set_speed = 9
controller.set_desired(set_speed)


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_pre = np.asarray(image)
        image_array = crop_img(image_pre)
        print('load img')
        #transformed_image_array = image_array[None, :, :, :]
        steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))
        # This model currently assumes that the features of the model are just the images. Feel free to change this.
        #steering_angle = 1.0*float(model.predict(transformed_image_array, batch_size=1))
        # The driving model currently just outputs a constant throttle. Feel free to edit this.
        throttle = controller.update(float(speed))
        print('pre ok')

        # smoothing by using previous steering angles
        #steering.append(steering_angle)
        #if len(steering) > 3:
        #    steering_angle = smoothing(steering, 3)
        #global num
        #cv2.imwrite('center_'+ str(num) +'.png', image_array)
        #num += 1
        #boost = 1 - speed/30.2 + 0.3
        #throttle = boost if (boost < 1) else 1
        #if abs(steering_angle) > 0.3:
            #throttle *= 0.2
        #throttle = 0.3
        print(steering_angle, throttle)
        #print("steering_angle : {:.3f}, throttle : {:.2f}".format(steering_angle, throttle))
        send_control(steering_angle, throttle)
       

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # check that model Keras version is same as local Keras version
   
    f = h5py.File(args.model, mode='r')

    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)
    
    json_string='model.json'
    with open(json_string, 'r') as jfile:#使用json保存模型结构
        model = model_from_json(json.load(jfile))
    print('load json')
    #model.compile("adam", "mse")
    weights_file = args.model
    #weights_file = args.model.replace('json', 'h5')
    
    model.load_weights(weights_file)
    print('load weight')
    """
    json_string='model.json'
    model = model_from_json(json_string)
    
    model = load_weights(args.model)
    """
    #model = load_model(args.model)
   

    

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

 













