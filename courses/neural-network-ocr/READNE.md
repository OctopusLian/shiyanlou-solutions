# 神经网络实现手写字符识别系统  

## 课程知识点  

1. 什么是神经网络  
2. 在客户端（浏览器）完成手写数据的输入与请求的发送  
3. 在服务器端根据请求调用神经网络模块并给出响应  
4. 实现BP神经网络  

## 系统构成  

+ 客户端（``ocr.js``）  
+ 服务器（``server.py``）  
+ 用户接口（``ocr.html``）  
+ 神经网络(``ocr.py``)  
+ 神经网络设计脚本(``neural_network_design.py``)  

用户接口(``ocr.html``)是一个``html``页面，用户在``canvans``上写数字，之后点击选择训练或是预测。客户端(``ocr.js``)将收集到的手写数字组合成一个数组发送给服务器端(``server.py``)处理，服务器调用神经网络模块(``ocr.py``)，它会在初始化时通过已有的数据集训练一个神经网络，神经网络的信息会被保存在文件中，等之后再一次启动时使用。最后，神经网络设计脚本(``neural_network_design.py``)是用来测试不同隐藏节点数下的性能，决定隐藏节点数用的。  

### 下载数据集

```
wget http://labfile.oss.aliyuncs.com/courses/593/data.csv
wget http://labfile.oss.aliyuncs.com/courses/593/dataLabels.csv
```

## 实验结果

输入``python server.py``打开服务器。在页面上写一个数字预测看看