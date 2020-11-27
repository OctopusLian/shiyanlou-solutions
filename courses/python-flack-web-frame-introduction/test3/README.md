## 静态文件及渲染模板  

### 知识点

- 静态文件
- 模板渲染

### 练习  

请创建一个模板和CSS文件，并在模板引入CSS文件，当访问网站首页时显示一个绿色的Hello ShiYanLou字样。  

### 参考答案  

文件目录结构如下所示：  

```
/hello.py
/templates
    /hello.html
/static
    /hello.css
```

hello.py 文件中的代码如下所示：  

```py
from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def get_hello():
    return render_template('hello.html')
```

templates/hello.html 存放的是前端代码，如下所示：  

```html
<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <link
      rel="stylesheet"
      type="text/css"
      href="{{ url_for('static', filename='hello.css') }}"
    />
  </head>
  <body>
    <h1>Hello ShiYanLou</h1>
  </body>
</html>
```

static/hello.css 存放的是 css 代码，如下所示：  

```css
h1 {
  color: green;
  text-align: center;
}
```