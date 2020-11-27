## 重定向、响应、会话和扩展  

### 知识点

- 重定向
- 响应
- 会话
- 扩展

### 练习  

请实现一个完整的用户登录功能:

- 当访问地址 http://127.0.0.1:5000/login ，出现登录页面，可以使用用户名和密码填写登录表单。
- 如果用户名和密码都为shiyanlou，那么就把用户名数据放到session中，把地址重定向到首页显示Hello shiyanlou，同时闪现消息 you were logged in。
- 如果用户名和密码不对，依然把地址重定向到首页显示hello world，同时闪现消息 username or password invalid。

### 参考答案  

文件目录结构如下所示：  

```
/app.py
/templates
    /index.html
    /login.html
```

在 app.py 文件中添加如下代码：  

```py
from flask import Flask, request
from flask import flash, redirect, url_for, render_template, session

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

@app.route('/<name>')
def index(name):
    return render_template('index.html',name=name)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['username'] != 'shiyanlou' or request.form['password'] != 'shiyanlou':
            flash('username or password invalid')
            return redirect(url_for('index', name='world'))
        else:
            session['username'] = request.form['username']
            name = request.form['username']
            flash('you were logged in')
            return redirect(url_for('index', name=name))
    return render_template('login.html')
```

在 templates/index.html 文件中添加如下代码：  

```html
<!DOCTYPE html>
<title>Index</title>
<div>
  {% for message in get_flashed_messages() %}
  <div class="flash">{{ message }}</div>
  {% endfor %}
  <h1>hello {{name}}</h1>
</div>
```

在 templates/login.html 文件中添加如下代码：  

```html
<!DOCTYPE html>
<title>Login</title>
<div>
  <form action="{{ url_for('login') }}" method="post">
    <dl>
      <dt>Username:</dt>
      <dd><input type="text" name="username" /></dd>
      <dt>Password:</dt>
      <dd><input type="password" name="password" /></dd>
      <dd><input type="submit" value="Login" /></dd>
    </dl>
  </form>
</div>
```
