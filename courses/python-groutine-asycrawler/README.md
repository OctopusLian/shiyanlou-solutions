# python实现基于协程的异步爬虫  

## 知识点  

1. 线程池实现并发爬虫  
2. 回调方法实现异步爬虫  
3. 协程技术的介绍  
4. 一个基于协程的异步编程模型  
5. 协程实现异步爬虫  

## 本地搭建测试网站  

1，我们使用 Python 2.7 版本官方文档作为测试爬虫用的网站  

```
    wget http://labfile.oss.aliyuncs.com/courses/574/python-doc.zip
    unzip python-doc.zip
```

这步已经做了，在本地的`res`文件夹中。  

2，安装``serve``，一个用起来很方便的静态文件服务器  

```
sudo npm install -g serve 
```

3，启动服务器  

```
serve python-doc
```
    
3-1，如果访问不了``npm``的资源，也可以用以下方式开启服务器  

```
ruby -run -ehttpd python-doc -p 3000
```

4，访问``localhost:3000``查看网站  

