## 路由介绍  

### 知识点

- 路由介绍  
- 变量规则  
- 重定向  
- 构建 URL  
- HTTP 方法  

### 练习  

- 请开发一个小应用，URL 地址输入http://127.0.0.1:5000/xxx（其中 xxx 表示你的名字），访问页面会显示 xxx。  
- 请完成一个应用，当 URL 是http://127.0.0.1:5000/sum/a/b时，其中a和b都是数字，服务器返回它们的和。  

### 参考答案  

练习题1  

```py
from flask import Flask
app = Flask(__name__)

@app.route('/<username>')
def get_name(username):
    return username
```

练习题2  

```py
from flask import Flask
app = Flask(__name__)

@app.route('/sum/<int:a>/<int:b>')
def get_sum(a,b):
    return '{0} + {1} = {2}'.format(a,b,a+b)
```
