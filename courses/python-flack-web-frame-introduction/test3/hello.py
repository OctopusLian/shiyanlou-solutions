from flask import Flask, render_template

app = Flask(__name__)

@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name=None):   # 默认 name 为 None
    return render_template('hello.html', name=name)   # 将 name 参数传递到模板变量中