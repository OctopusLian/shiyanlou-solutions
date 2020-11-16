# 高德API+Python解决租房问题  

## 课程知识点  

本课程项目完成过程中，我们将学习：  

1. ``requests``、``BeautifulSoup``、``csv`` 等库的简单使用  
2. 高德地图 Javascript API 的使用  

## 项目文件说明  

实验中会用到三个文件：
``rent.csv``，由``crawl.py``生成，是房源文件。  
``crawl.py``是一个非常简单的爬取网页的脚本。
``index.html``是最重要的显示地图的部分。  

## 演示  

输入``python -m SimpleHTTPServer 3000``打开服务器，浏览器访问``localhost:3000``查看效果：

首先选择工作地点：

![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid8834labid1978timestamp1470291985646.png/wm)

划出了一小时内的通勤范围：

![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid8834labid1978timestamp1470292012045.png/wm)

北京堵车太猖狂，可能还是地铁保险：

![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid8834labid1978timestamp1470292051292.png/wm)

导入房源文件：

![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid8834labid1978timestamp1470292226208.png/wm)

导入后：

![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid8834labid1978timestamp1470292244419.png/wm)


选择一处房源，会自动帮你规划路径：

![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid8834labid1978timestamp1470292326683.png/wm)

选中房源地址跳转到目标页面：

![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid8834labid1978timestamp1470292416181.png/wm)