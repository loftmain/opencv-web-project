from flask import Flask, request, render_template, redirect
import cv2
import numpy as np
import time

image = None
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
    global image

    f = request.files["file1"]
    filename = "./static/" + f.filename
    f.save(filename)

    image = cv2.imread(filename)
    cv2.imwrite("./static/result.jpg", image)
    print(image.shape)

    return redirect("/")

@app.route('/imageprocess')
def imageprocess():
    global image
    method = request.args.get("method")
    if method == "emboss":
        # 엠보싱 연산 opencv code
        # save result.jpeg 로 항상 static 볼더에 할 수 있음. /static/result.jpg
        emboss = np.array([
            [-1, -1, 0],
            [-1, 0, 1],
            [0, 1, 1]], np.float32)

        dst = cv2.filter2D(image, -1, emboss, delta=128)
        filename = "./static/" + "result.jpg"
        cv2.imwrite(filename, np.hstack((image, dst)))

    if method == "blur":
        size = int(request.args.get("size", 3))
        dst = cv2.blur(image, (size,size))
        cv2.imwrite("./static/result.jpg", np.hstack((image, dst)))

    if method == "sharp" :
        sharp = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]])

        dst = cv2.filter2D(image, -1, sharp)
        filename = "./static/" + "result.jpg"
        cv2.imwrite(filename, np.hstack((image, dst)))
        # image = cv2.imread(filename)


    return "hello~~"

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8000)
