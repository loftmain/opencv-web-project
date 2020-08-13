import cv2
import numpy as np
from pyzbar.pyzbar import decode
import io

from flask import Flask, request, render_template, make_response, Response
app = Flask(__name__)

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')
vc = cv2.VideoCapture(0)

@app.route('/')
def index():
    return  render_template("face.html")

@app.route('/upload', methods=["post"])
def upload():
    f = request.files['file1']   
    filename = "./static/" + f.filename
    f.save(filename)            
    #img = cv2.imdecode(np.fromstring(f.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.imread(filename)    
    #img = cv2.resize(img, dsize=(0, 0), fx=0.25, fy=0.25)        
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)
    
    if len(faces) > 0 :   
        for (x1, y1, w1, h1) in faces:
            cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 255), 2)

            faceROI = gray[y1:y1 + h1, x1:x1 + w1]
            cv2.imshow('face', faceROI)                

            eyes = eye_classifier.detectMultiScale(faceROI)
            for (x2, y2, w2, h2) in eyes:
                center = (int(x1+ x2 + w2 / 2), int(y1 +y2 + h2 / 2))
                cv2.circle(img, center, int(w2 / 2), (255, 0, 0), 2, cv2.LINE_AA)

        retval, buffer = cv2.imencode('.jpg', img)  


        response = make_response(buffer.tobytes())
        response.headers['Content-Type'] = 'image/jpg'

        return response    
    
    code = decode(img)    
    url  = code[0].data.decode("utf-8")    
    
    return f"<a href={url}>QR코드가기</a>" 
    
    #return  f"<script> alert('이동합니다.'); window.location.href='{url}'; </script>"
    
    
    
def gen():
    while True:
        read_return_code, frame = vc.read()
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

@app.route('/test')
def test():
    return  "<h1>동영상 테스트</h1> <img src=/video_feed width=320 height=240>"



if __name__ == '__main__':
     app.run(host='0.0.0.0', debug=True, port=8000)      
