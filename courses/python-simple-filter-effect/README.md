# Python 实现简单滤镜  

## 课程知识点

- 使用 docopt 构建命令行解析器  
- 使用 struct 模块解析 ACV 格式文件  
- Pillow 图像操作  

## 环境配置信息

- Python：3.5.2  
- Numy：1.11.2  
- Scipy：0.18.1  
- Pillow：3.4.2  

## 编程实现  

1.需要构建命令行解析器从命令中解析出文件路径参数  
2.加载图像与滤镜文件  
3.处理图像  
4.保存处理后的图像  

## 运行方式  

```
$ python3 filter.py <curves> <image>
```

## 参考链接  

[docopt官方文档](http://docopt.org/)  
[ACV 文件的数据格式](https://www.adobe.com/devnet-apps/photoshop/fileformatashtml/#50577411_pgfId-1056330)  
[struct — Interpret bytes as packed binary data](https://docs.python.org/3/library/struct.html?highlight=struct#module-struct)  
[scipy.interpolate.lagrange](https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.interpolate.lagrange.html#scipy.interpolate.lagrange)  
[numpy.poly1d](https://numpy.org/doc/stable/reference/generated/numpy.poly1d.html)