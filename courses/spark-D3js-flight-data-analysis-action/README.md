# Spark 和 D3.js 分析航班大数据  
https://www.lanqiao.cn/courses/610  

## 实验内容  

>“我们很抱歉地通知您，您乘坐的由 XX 飞往 XX 的 XXXX 航班延误。”  

相信很多在机场等待飞行的旅客都不愿意听到这句话。随着乘坐飞机这种交通方式的逐渐普及，航延延误问题也一直困扰着我们。航班延误通常会造成两种结果，一种是航班取消，另一种是航班晚点。  

在本课程中，我们将通过 Spark 提供的 DataFrame、 SQL 和机器学习框架等工具，基于 D3.js 数据可视化技术，对航班起降的记录数据进行分析，尝试找出造成航班延误的原因，以及对航班延误情况进行预测。  

## 实验知识点  

Spark DataFrame 操作  
Spark SQL 常用操作  
Spark MLlib 机器学习框架使用  

## 实验环境  

Xfce 终端  
Apache Spark 2.1.1  
D3.js  
文本编辑器 gedit  
FireFox 浏览器  

## 开发准备  

### 数据集简介及准备  

本节实验用到的航班数据集仍然是 2009 年 Data Expo 上提供的飞行准点率统计数据。  

此次我们选用 1998 年的数据集。你可以通过官方下载链接来下载，也可以获取实验楼为你提供的副本。  

>如果你是在自己的 Spark 集群上进行学习，则可以选用 2007、2008 年等年份的数据集。它们含有的航班数量更多，能够得到更多的信息。  

```
wget https://labfile.oss.aliyuncs.com/courses/610/1998.csv.bz2
```


## OpenRefine  
 是 Google 主导开发的一款开源数据清洗工具。我们先在环境中安装它：  

 ```
 wget https://labfile.oss.aliyuncs.com/courses/610/openrefine-linux-3.2.tar.gz
tar -zxvf openrefine-linux-3.2.tar.gz
cd openrefine-3.2
# 启动命令
./refine
 ```

当出现下图所示的提示信息后，在浏览器中打开 URL http://127.0.0.1:3333/。  

Open Refine 启动成功的标志是出现 Point your browser to http://127.0.0.1:3333 to start using Refine 的提示  

