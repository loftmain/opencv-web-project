from flask import Flask, request, render_template, redirect
import cv2
import numpy as np
import time

app = Flask(__name__)

def chromakey_background(img, background):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    patch = hsv[0:20, 0:20, :]

    # 범위를 조금 넓힌다.
    minH = np.min(patch[:,:,0])*0.9 # 90%
    maxH = np.max(patch[:,:,0])*1.1 # 110%

    minS = np.min(patch[:,:,1])*0.9
    maxS = np.max(patch[:,:,1])*1.1

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    dest1 = img.copy()
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            if h[r,c] >=minH and h[r,c] <=maxH and \
                s[r,c] >minS and s[r,c] <=maxS:
                dest1[r, c, :] = background[r, c, :]
            else:
                dest1[r, c, :] = img[r, c, :]
    return dest1

@app.route('/')
def index():
    return render_template("imageprocessing.html", ctx={"title":"영상처리"})

@app.route('/upload', methods=["post"])
def upload():
    f = request.files["file1"]
    filename = "./static/" + f.filename
    f.save(filename)

    f1 = request.files["file2"]
    filename1 = "./static/" + f1.filename
    f1.save(filename1)

    img = cv2.imread(filename)
    img = cv2.resize(img, dsize=(320, 240))

    background = cv2.imread(filename1)
    background = cv2.resize(background, dsize=(320, 240))

    img = chromakey_background(img, background)
    cv2.imwrite(filename, img)

    now = time.localtime()
    return redirect("/")

@app.route('/imageprocess')
def imageprocess():

    method = request.args.get("method")
    if method == "emboss":
        # 엠보싱 연산 opencv code
        # save result.jpeg 로 항상 static 볼더에 할 수 있음. /static/result.jpg
        pass
    return "hello~~"

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8000)
