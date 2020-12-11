## 1.1 实验内容

本节课程介绍 Selenium 的功能作用及安装、环境配置，并介绍 Selenium 常用的语法。  

## 1.2 实验知识点

Selenium 介绍  
安装 Selenium  
安装 geckodriver 浏览器驱动  
Selenium 的元素定位  
点击元素  
清空文本输入框、向文本输入框输入文本  
获取元素属性  
下拉页面  
页面弹窗的定位以及弹窗文本的获取  
窗口跳转  
iframe 定位  

## 步骤  

### geckodriver  

既然名为网页浏览器自动化自然要安装浏览器，一般来说，Chrome、Firefox等浏览器都可以，这里我们使用当前系统自带的Firefox作为实验浏览器。  

现在我们需要下载对应的浏览器驱动geckodriver，在xfce中输入以下命令：  

```
$ wget https://labfile.oss.aliyuncs.com/courses/1163/geckodriver-v0.22.0-linux64.tar.gz

$ tar zxvf geckodriver-v0.22.0-linux64.tar.gz
$ sudo mv geckodriver /usr/local/bin
```

下面我们来验证是否正常安装，在终端使用命令vim demo.py创建文件并写入代码：  

```py
#! /usr/bin/python3

from selenium import webdriver

driver = webdriver.Firefox()
driver.get("https://www.lanqiao.cn")
```