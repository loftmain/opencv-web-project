
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def index():
    html = """
    <h1>안녕하세요<h1>
    """
    return html

datas = [45.7, 45, 10, 70.8]

@app.route('/signal')
def signal():
    global datas # global 변수인 것 명시
    datas.append( float(request.args.get("data")) )
    return str(datas)

@app.route('/view')
def view():
    global datas # global 변수인 것 명시
    
    labels = [i+1 for i in range(len(datas))]
                              
    ctx = {"title":"그래프^*^", "labels":labels, "data":datas}
    return render_template("view.html", ctx=ctx) # 반드시 ctx 딕셔너리로 보내기


if __name__== '__main__':
    app.run(host='0.0.0.0', debug=True, port=8000)
