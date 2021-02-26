# 使用OpenCV&&C++进行模板匹配  

## 一、课程介绍  

### 1.学习目标  

- 学会用imread载入图像，和imshow输出图像。  
- 用nameWindow创建窗口，用createTrackbar加入滚动条和其回调函数的写法。  
- 熟悉OpenCV函数matchTemplate并学会通过该函数实现模板匹配。  
- 学会怎样将一副图片中自己感兴趣的区域标记出来  

### 2.什么是模板匹配？  

在一副图像中寻找和另一幅图像最相似（匹配）部分的技术。  

### 3.案例展示  

输入两张图像分别为  

- template.jpg 

![](http://cfwfs.img48.wal8.com/img48/559876_20161021194215/148024093552.jpg)  
- original.jpg

![](http://cfwfs.img48.wal8.com/img48/559876_20161021194215/148024084784.jpg)  

以上两张图片匹配完成的输出结果图片  

- result.jpg  

![](http://cfwfs.img48.wal8.com/img48/559876_20161021194215/148024086031.jpg)  

## 二、实验原理  

让模板图片在原图片上的一次次滑动（从左到右，从上到下一个像素为单位的移动），然后将两张图片的像素值进行比对，然后选择相似度最高的部分进行标记，当遇到相似度更高的部分时更换标记部分。扫描完毕之后，将相似度最高的部分标记出来，作为图片进行输出操作。  

## 三、环境搭建  

* 本实验需要先在实验平台安装 OpenCV ，需下载依赖的库、源代码并编译安装。安装过程建议按照教程给出的步骤，或者你可以参考官方文档中 Linux 环境下的[安装步骤](http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html)，但 **有些选项需要变更**。安装过程所需时间会比较长，这期间你可以先阅读接下来的教程，在大致了解代码原理后再亲自编写尝试。  

* 我提供了一个编译好的`2.4.13-binary.tar.gz`包，你可以通过下面的命令下载并安装，节省了编译的时间，通过这个包安装大概需要20～30分钟，视实验楼当前环境运转速度而定。  

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

* 如果你想体验编译的整个过程，我也提供了一个一键安装的脚本文件，你可以通过下面的命令尝试。这个过程会非常漫长，约2小时，期间可能还需要你做一定的交互确认工作。  

```bash
$ cd ~
$ sudo apt-get update
$ wget http://labfile.oss.aliyuncs.com/courses/671/opencv.sh
$ sudo chmod 777 opencv.sh
$ ./opencv.sh
```

* 如果你觉得有必要亲自尝试一下安装的每一步，可以按照下面的命令逐条输入执行，在实验楼的环境中大概需要两个小时。  

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

* 检验配置是否成功。将 OpenCV 自带的例子（在目录`PATH_TO_OPENCV/samples/C`下）运行检测。如果成功，将显示 lena 的脸部照片，同时圈出其面部。  

```bash
$ cd samples/C
$ ./build_all.sh
$ ./facedetect --cascade="/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml" --scale=1.5 lena.jpg
```

## 四、实验步骤  

### 1.定义头文件  

在这里我们用了
```
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
```

这三个头文件：  

`highgui.hpp`：定义了创建窗口的flag，窗口事件的flag，Qt窗口的flag，事件回调函数原型，以及窗口/控件操作相关的系列函数，openGL的包装函数；图像输入输出显示的相关函数；视频摄像头输入输出显示的相关函数，VideoCapture，VideoWriter。  

`imgproc.hpp`：定义了图像处理模块之平滑和形态学操作。  

`iostream`：不再赘述。  

### 2.设计主要功能并对其解析（从main函数入口开始分析）  

**imread函数**：  

imread函数可以将图片读取然后放到Mat容器里面用于后续操作。  

```
img = imread("original.jpg");//载入元图像
templ = imread("template.jpg");//载入模版图像
```

**nameWindow函数**：  

1. 创建窗口。第一个参数是窗口名称，第二个窗口是int类型的Flag可以填写以下的值  
2. WINDOW_NORMAL设置了这个值，用户便可以改变窗口的大小（没有限制）  
3. WINDOW_AUTOSIZE如果设置了这个值，窗口大小会自动调整以适应所显示的图像，并且不能手动改变窗口大小  

```
namedWindow( image_window, CV_WINDOW_AUTOSIZE ); // 窗口名称，窗口标识CV_WINDOW_AUTOSIZE是自动调整窗口大小以适应图片尺寸。
namedWindow( result_window, CV_WINDOW_AUTOSIZE );
```
**createTrackba函数**：  

创建滑动条，第一个参数是匹配方法，第二个参数是确定滑动条所在窗口，第三个参数是对应滑动条的值，第四个参数是滑动条的最大值，第五个参数是回调函数。  

```
//创建滑动条
createTrackbar("匹配方法", image_window, &match_method, max_Trackbar, MatchingMethod ); //滑动条提示信息，滑动条所在窗口名，匹配方式（滑块移动之后将移动到的值赋予该变量），回调函数。
```

**自己写的回调函数**  

先调用回调函数，在没有滑动滑块的时候也有图像。  

```
MatchingMethod( 0, 0 );//初始化显示
```
**waitkey函数**：  

其取值可以是<=0或大于0.当取值为<=0的时候，如果没有键盘触发则一直等待，否则返回值为按下去的ascll对应数字。  

```
waitKey(0); //等待按键事件，如果值0为则永久等待。
```
**Mat::copyto函数**：  

创建Mat类型数据结构img_display。并将img内容赋值给img_display。  

```
Mat img_display;
img.copyTo( img_display ); //将 img的内容拷贝到 img_display
```

**Mat::create函数**：  

计算用于储存结果的输出图像矩阵大小并创建所需的矩阵。  

```
//创建输出结果的矩阵
int result_cols =  img.cols - templ.cols + 1;     //计算用于储存匹配结果的输出图像矩阵的大小。
int result_rows = img.rows - templ.rows + 1;
result.create( result_cols, result_rows, CV_32FC1);//被创建矩阵的列数，行数，以CV_32FC1形式储存。
```
**matchTemplate （模版匹配）函数** :  

我们在createTrackba函数那里见到过match_method变量，这个是决定匹配方法的变量，由滑块确定。  

```
matchTemplate( img, templ, result, match_method ); //待匹配图像，模版图像，输出结果图像，匹配方法（由滑块数值给定。）
```
**normalize（归一化函数）**:  

归一化就是要把需要处理的数据经过处理后（通过某种算法）限制在你需要的一定范围内。首先归一化是为了后面数据处理的方便，其次是保证程序运行时收敛加快。归一化的具体作用是归纳统一样本的统计分布性。归一化在0-1之间是统计的概率分布，归一化在某个区间上是统计的坐标分布。归一化有同一、统一和合一的意思。  

```
normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );//输入数组，输出数组，range normalize的最小值，range normalize的最大值，归一化类型，当type为负数的时候输出的type和输入的type相同。
```

**minMaxLoc函数** :  

用于寻找距震中的最大值和最小值  

```
minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );//用于检测矩阵中最大值和最小值的位置
```

**不同方法之间选择最佳精确度**:  

对于方法CV_TM_SQDIFF，和CV_TM_SQDIFF_NORMED，越小的数值代表越准确匹配结果，而对于其他方法，数值越大匹配的准确度越高。  

``` 
if( match_method  == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED )
{ matchLoc = minLoc; }
else
{ matchLoc = maxLoc; }
```

**将最后得到的结果显性的标记出来**  

第一个参数（img）：将要被操作的图像，第二个和第三个参数分别是一个矩形的对角点。第四个（color）参数是线条的颜色（RGB）。第五个参数（thickness）：组成矩阵线条的粗细程度。第六个参数（line_type）：线条的类型，见cvLine的描述。第七个参数shift：坐标点的小数点位数  

```
//让我看看您的最终结果
rectangle( img_display, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar(0,0,255), 2, 8, 0 ); //将得到的结果用矩形框起来
rectangle( result, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar(0,0,255), 2, 8, 0 );
```
### 3.应用算法解析  

**matchTemplate实现了末班匹配散发：其中可用的方法有六个：**  

1.平方差匹配： method = CV_TM_SQDIFF  
- 这类方法利用平方差来进行匹配最好匹配为0.匹配差越大，匹配值越大。  

![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid243326labid2379timestamp1480338580748.png/wm)  

2.标准平方差匹配：method = CV_TM_SQDIFF_NORMED  
- 这类方法利用平方差来进行匹配最好匹配为0.匹配差越大，匹配值越大。  

![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid243326labid2379timestamp1480338680280.png/wm)  

3.相关匹配method=CV_TM_CCORR  
- 这类方法采用模板和图像间的乘法操作,所以较大的数表示匹配程度较高,0标识最坏的匹配效果.  

![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid243326labid2379timestamp1480338742541.png/wm)  

4.标准相关匹配 method=CV_TM_CCORR_NORMED  
- 同标准平方差和平方差，以下不再赘述。  

![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid243326labid2379timestamp1480338785059.png/wm)  

5.相关匹配 method=CV_TM_CCOEFF  
- 这类方法将模版对其均值的相对值与图像对其均值的相关值进行匹配,1表示完美匹配,-1表示糟糕的匹配,0表示没有任何相关性(随机序列).  

![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid243326labid2379timestamp1480338852976.png/wm)  

6.标准相关匹配 method=CV_TM_CCOEFF_NORMED  
- 通常,随着从简单的测量(平方差)到更复杂的测量(相关系数),我们可获得越来越准确的匹配(同时也意味着越来越大的计算代价). 最好的办法是对所有这些设置多做一些测试实验,以便为自己的应用选择同时兼顾速度和精度的最佳方案.  

![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid243326labid2379timestamp1480338896621.png/wm)  

## 五、实验程序  

** 这里就是完整的代码，上面对这些代码已经做了完整的解析。相信你已经可以看懂下方代码了。**  

```cpp
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>


using namespace std;
using namespace cv;

Mat img; Mat templ; Mat result;


int match_method;
int max_Trackbar = 5;




void MatchingMethod( int, void* )
{

  Mat img_display;
  img.copyTo( img_display ); //将 img的内容拷贝到 img_display

  /// 创建输出结果的矩阵
  int result_cols =  img.cols - templ.cols + 1;     //计算用于储存匹配结果的输出图像矩阵的大小。
  int result_rows = img.rows - templ.rows + 1;

  result.create( result_cols, result_rows, CV_32FC1 );//被创建矩阵的列数，行数，以CV_32FC1形式储存。

  /// 进行匹配和标准化
  matchTemplate( img, templ, result, match_method ); //待匹配图像，模版图像，输出结果图像，匹配方法（由滑块数值给定。）
  normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );//输入数组，输出数组，range normalize的最小值，range normalize的最大值，归一化类型，当type为负数的时候输出的type和输入的type相同。

  /// 通过函数 minMaxLoc 定位最匹配的位置
  double minVal; double maxVal; Point minLoc; Point maxLoc;
  Point matchLoc;

  minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );//用于检测矩阵中最大值和最小值的位置

  /// 对于方法 SQDIFF 和 SQDIFF_NORMED, 越小的数值代表更高的匹配结果. 而对于其他方法, 数值越大匹配越好
  if( match_method  == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED )
    { matchLoc = minLoc; }
  else
    { matchLoc = maxLoc; }

  /// 让我看看您的最终结果
  rectangle( img_display, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar(0,0,255), 2, 8, 0 ); //将得到的结果用矩形框起来
  rectangle( result, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar(0,0,255), 2, 8, 0 );

  imshow( "Source Image", img_display );//输出最终的到的结果
  imwrite("result.jpg",img_display); //将得到的结果写到源代码目录下。
  imshow( "Result window", result );   //输出匹配结果矩阵。

  return;
}


int main( int argc, char** argv )
{

  img = imread("original.jpg");//载入待匹配图像
  templ = imread("template.jpg");//载入模版图像

  /// 创建窗口
  namedWindow( "Source Image", CV_WINDOW_AUTOSIZE ); // 窗口名称，窗口标识CV_WINDOW_AUTOSIZE是自动调整窗口大小以适应图片尺寸。
  namedWindow( "Result window", CV_WINDOW_AUTOSIZE );

  /// 创建滑动条
  createTrackbar("jackchen", "Source Image", &match_method, max_Trackbar, MatchingMethod ); //滑动条提示信息，滑动条所在窗口名，匹配方式（滑块移动之后将移动到的值赋予该变量），回调函数。

  MatchingMethod( 0, 0 );//初始化显示

  int logo = waitKey(5000); //等待按键事件，如果值0为则永久等待。

  return 0;
}
```
## 六、实验结果
执行命令
```
wget http://labfile.oss.aliyuncs.com/courses/716/template.jpg
wget http://labfile.oss.aliyuncs.com/courses/716/original.jpg
```
将两幅输入图片template.jpg 和 original.jpg 下载到存放源代码main.cpp的文件夹，执行
```
g++ -ggdb `pkg-config --cflags opencv` -std=c++11 -fpermissive -o `basename main` main.cpp `pkg-config --libs opencv`
```
对main.cpp进行编译，编译成功后目录下将产生一个名为 main的可执行文件，在终端键入如下命令
```
sudo vim /etc/ld.so.conf.d/opencv.conf
```
写入内容
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib  
```
更新然后运行
```
sudo ldconfig -v
./main
```

对main.cpp进行编译并执行程序，在文件夹下会多出一个名为`result.jpg`的图片，这就是匹配出来的结果图片。
