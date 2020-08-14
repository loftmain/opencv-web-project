import cv2
import numpy as np
import io
import random

from flask import Flask, request, render_template, make_response, Response
app = Flask(__name__)

vc = cv2.VideoCapture('vtest.avi')

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

datas = []

def detectHuman(frame) :
    detected, _ = hog.detectMultiScale(frame)

    for (x, y, w, h) in detected:
        c = (random.randint(0, 255),
             random.randint(0, 255), 
             random.randint(0, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), c, 3)
    return detected    

@app.route('/')
def index():
    return  render_template("human.html")

def gen():
    global vc
    global datas
    
    while True:
        read_return_code, frame = vc.read()
        
        if not read_return_code :
            vc = cv2.VideoCapture('vtest.avi')
            datas = []
            break;
        
        rect = detectHuman(frame)
        
        datas.append(len(rect))
        print(datas)
        
        encode_return_code, image_buffer = cv2.imencode('.jpg', frame)
        io_buf = io.BytesIO(image_buffer)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + io_buf.read() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(
        gen(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/view')
def view():
    global datas
    return str(datas)
    
if __name__ == '__main__':
     app.run(host='0.0.0.0', debug=True, port=8000)      
