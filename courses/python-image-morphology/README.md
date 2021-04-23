# Python从零开始实现图像形态学操作  

## 实验介绍  

本实验不依赖 OpenCV 等图形处理库，从矩阵操作开始逐步实现形态学基本算子(腐蚀、扩张、开、闭)，并使用这些算子处理二值图像和灰度图像。待处理的灰度图像如下图所示：  

## 实验知识点  

图像存储和显示原理  
Python Numpy 矩阵操作  
图像灰度阈值变换  
二值图像和灰度图像的腐蚀运算  
二值图像和灰度图像的膨胀运算  
二值图像和灰度图像的开运算  
二值图像和灰度图像的闭运算  

## 实验环境  

Python 3  
Numpy 1.14  
PIL 5.2  

## 开发准备  

```
$ sudo apt-get update
$ sudo apt-get install python3-pip
$ sudo pip3 install numpy
$ sudo apt-get install libjpeg-dev
$ sudo pip3 install pillow
```