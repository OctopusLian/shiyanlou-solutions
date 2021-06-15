# Python3 实现 Markdown 解析器  

## 课程简介  

通过本次课程的学习，我们将接触到以下知识点：  

正则表达式  
docopt 构建命令行解析器  
简单的 HTML 语法  

## 执行  

```sh
$ python3 md2pdf.py doc_template.md -p template.pdf
```

## QA  

Q1：使用 wkhtmltopdf 这个工具将在线的网页转化成 PDF 打印出来。  

A1：使用命令 wkhtmltopdf [url] [outputfile] 即可完成打印。比如 wkhtmltopdf https://www.shiyanlou.com shiyanlou.pdf  