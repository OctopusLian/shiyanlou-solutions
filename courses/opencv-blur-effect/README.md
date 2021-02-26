# 使用 OpenCV 进行图片平滑处理打造模糊效果  

## 介绍  

### 1.学习目标
- 学习使用归一化滤波器，高斯滤波器，双边滤波器并懂得其相关数学知识。  
- 图片中“核”的意义，与如何用核实现卷积。  
- 熟悉OpenCV函数filter2D并学会通过该函数实现线性模糊。  
- 学习使用blur，GaussianBlur，medianBlur，bilateralFilter来对图片进行平滑操作。  

### 2.什么是图像平滑/模糊？  
- 平滑也称为模糊，是一项简单且使用频率很高的图像处理方法。  
- 平滑的处理用途很多，在这里我们只关注减少噪声和模糊操作。  
- 对图像进行平滑/模糊操作之后就像近视的人没戴眼镜看一副图片的感觉。  

### 3.案例展示  
输入一张图像：  
- original.jpg 

![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid18510labid2394timestamp1481595981170.png/wm)  

输出一张图像：
- result.jpg   
![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid18510labid2394timestamp1481598537751.png/wm)  

## 二、实验原理  

平滑处理的时候我们需要用到一个滤波器。最常用的滤波器是线性滤波器，线性滤波器处理的输出像素值(i.e. g(i,j))是输入像素值 (i.e. f(i+k,j+l))的加权和 :  
![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/ document-uid18510labid2394timestamp1481614244221.png/wm)  
可以将滤波器想象成一个包含加权系数的窗口，当使用这个滤波器平滑处理图像的时候，就把这个窗口滑过图像。  

## 三、环境搭建  

本实验需要先在实验平台安装 OpenCV ，需下载依赖的库、源代码并编译安装。安装过程建议按照教程给出的步骤，或者你可以参考官方文档中 Linux 环境下的[安装步骤](http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html)，但 **有些选项需要变更**。安装过程所需时间会比较长，这期间你可以先阅读接下来的教程，在大致了解代码原理后再亲自编写尝试。  

我提供了一个编译好的`2.4.13-binary.tar.gz`包，你可以通过下面的命令下载并安装，节省了编译的时间，通过这个包安装大概需要20～30分钟，视实验楼当前环境运转速度而定。  

```bash
$ sudo apt-get update
$ sudo apt-get install build-essential libgtk2.0-dev libjpeg-dev libtiff5-dev libjasper-dev libopenexr-dev cmake python-dev python-numpy python-tk libtbb-dev libeigen2-dev yasm libfaac-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev libqt4-dev libqt4-opengl-dev sphinx-common texlive-latex-extra libv4l-dev libdc1394-22-dev libavcodec-dev libavformat-dev libswscale-dev
$ cd ~
$ mkdir OpenCV && cd OpenCV
$ wget http://labfile.oss.aliyuncs.com/courses/671/2.4.13-binary.tar.gz
$ tar -zxvf 2.4.13-binary.tar.gz
$ cd opencv-2.4.13
$ cd build
$ sudo make install
```

`特别说明：`以下两种环境搭建方式耗时较长，大家可以在本地环境进行尝试，实验楼的课程环境暂时可能会等待很久。  

如果你想体验编译的整个过程，我也提供了一个一键安装的脚本文件，你可以通过下面的命令尝试。这个过程会非常漫长，约2小时，期间可能还需要你做一定的交互确认工作。  

```bash
$ cd ~
$ sudo apt-get update
$ wget http://labfile.oss.aliyuncs.com/courses/671/opencv.sh
$ sudo chmod 777 opencv.sh
$ ./opencv.sh
```

如果你觉得有必要亲自尝试一下安装的每一步，可以按照下面的命令逐条输入执行，在实验楼的环境中大概需要两个小时。  

```bash
$ sudo apt-get update
$ sudo apt-get install build-essential libgtk2.0-dev libjpeg-dev libtiff5-dev libjasper-dev libopenexr-dev cmake python-dev python-numpy python-tk libtbb-dev libeigen2-dev yasm libfaac-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev libqt4-dev libqt4-opengl-dev sphinx-common texlive-latex-extra libv4l-dev libdc1394-22-dev libavcodec-dev libavformat-dev libswscale-dev
$ wget https://github.com/Itseez/opencv/archive/2.4.13.zip
$ unzip 2.4.13.zip
$ cd 2.4.13
$ mkdir release && cd release
$ cmake -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D WITH_QT=ON -D WITH_GTK=ON -D WITH_OPENGL=ON ..
$ sudo make
$ sudo make install
$ sudo gedit /etc/ld.so.conf.d/opencv.conf   
$ 输入 /usr/local/lib，按 Ctrl + X 退出，退出时询问是否保存，按 Y 确认。
$ sudo ldconfig -v
$ sudo gedit /etc/bash.bashrc
$ 在文件末尾加入
$ PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig
  export PKG_CONFIG_PATH
  按 Ctrl + X 退出，按 Y 确认保存。
```

检验配置是否成功。将 OpenCV 自带的例子（在目录`PATH_TO_OPENCV/samples/C`下）运行检测。如果成功，将显示 lena 的脸部照片，同时圈出其面部。  

```bash
$ cd samples/C
$ ./build_all.sh
$ ./facedetect --cascade="/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml" --scale=1.5 lena.jpg
```

## 对四种滤波器的介绍  

#### **归一化滤波器**  

这是最简单的滤波器输出的值为核窗口内像素值的均值。公式如下：  

![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid18510labid2394timestamp1481614816157.png/wm)

#### **高斯滤波器**  

这是最有用也是最常用的滤波器虽然他的速度不是很快，高斯滤波是将输入数组的每一个像素点，核高斯内核进行卷积讲卷积核输出做像素值。下面是一维高斯函数图像可以发现周围像素的加权系数随着距离中心的距离越来越远，而变得越来越小。这个是一维的，当然二维图片也是这个原理的。  

![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid18510labid2394timestamp1481614576035.png/wm)

在二维图片中高斯函数的公式如下：
![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid18510labid2394timestamp1481614644531.png/wm)其中 ![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid18510labid2394timestamp1481614686471.png/wm)为均值 (峰值对应位置)， ![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid18510labid2394timestamp1481614715981.png/wm)代表标准差 (变量 x 和 变量 y 各有一个均值，也各有一个标准差)  

#### **中值滤波器**  

以当前像素为中心的正方形区域内所有像素值的中值作为当前像素的像素值。  

#### **双边滤波**  

 目前我们了解的滤波器都是为了平滑图像， 问题是有些时候这些滤波器不仅仅削弱了噪声， 连带着把边缘也给磨掉了。 为避免这样的情形 (至少在一定程度上 ), 我们可以使用双边滤波。  
 类似于高斯滤波器，双边滤波器也给每一个邻域像素分配一个加权系数。 这些加权系数包含两个部分, 第一部分加权方式与高斯滤波一样，第二部分的权重则取决于该邻域像素与当前像素的灰度差值。  

