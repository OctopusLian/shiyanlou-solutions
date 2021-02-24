# 制作Markdown预览器  

## 代码结构  

克隆完成以后`golang-markdown-previewer`的目录结构如下：  

```
golang-markdown-previewer
├── README.md
└── src
    ├── previewer
    │   ├── http_server.go
    │   ├── md_converter.go
    │   ├── previewer.go
    │   ├── template.go
    │   ├── watcher.go
    │   └── websocket.go
    └── sysm
        ├── http_server.go
        ├── sysm.go
        ├── template.go
        ├── watcher.go
        └── websocket.go

```

## 实验原理  

### 2.1 Markdown 预览器的设计  

本项目课中，我们将使用 [`go 语言`](http://www.shiyanlou.com/courses/11)编写一个 markdown 文件的实时预览器，它可以在浏览器实时预览使用任何文本编辑器正在编辑的 markdown 文件。  

什么是 markdown 呢？  

> Markdown 是一种轻量级标记语言，创始人为约翰 · 格鲁伯（John Gruber）。它允许人们 “使用易读易写的纯文本格式编写文档，然后转换成有效的 XHTML(或者 HTML) 文档”。  

我们可以使用 markdown 编写纯文本的文件，然后通过软件将这些文本格式化为排版优美的 HTML 页面。实际上，本课程就是使用 markdown 进行编写。  

预览器的工作原理是什么呢？其实非常简单：预览器会监控 markdown 文件的状态，如果检测到发生变化就将 markdown 文件格式化为 html 页面重新显示到浏览器上。所以，我们的预览器将包含：  

- http 服务器：用于显示文本  
- markdown 转换器： 将 markdown 文件转换为 html 页面  

除此以外，为了实时预览，我们将使用 websocket 技术。  

### 2.2 Websocket 浅析  

我们的预览器会在浏览器中实时预览我们编辑的 markdown 文件。在浏览器中实现实时的响应，有两种方式，第一种是通过浏览器的轮询方式，浏览器端不断的向服务端请求数据（主要通过客户端 javascript），然后更新页面上的数据。不过如今我们可以使用 websocket 解决这类问题了，利用 websocket 可以在浏览器和服务器建立一个全双工的通道，这样服务端可以直接将新的数据发送给浏览器，浏览器在页面上更新这些数据即可。  

什么是 Websocket？  

> WebSocket 是 HTML5 开始提供的一种浏览器与服务器间进行全双工通讯的网络技术。WebSocket 通信协议于 2011 年被 IETF 定为标准 RFC 6455，WebSocketAPI 被 W3C 定为标准。在 WebSocket API 中，浏览器和服务器只需要做一个握手的动作，然后，浏览器和服务器之间就形成了一条快速通道。两者之间就直接可以数据互相传送。  

#### 2.2.1 Websocket 的协议转换  

Websocket 是工作在 http 协议之上的，我们都知道 http 协议是无状态的，那浏览器和服务器是怎么样知道将 http 协议转换为 websocket 协议的呢？  

在使用 wbesocket 的时候，浏览器向服务发送一个请求，表明它要将协议由 http 转为 websocket。客户端通过 http 头中的 Upgrade 属性来表达这一请求，下面是一个请求 http 头的示例:  

```
GET ws://localhost:6060/1.md HTTP/1.1
Host: localhost:6060
Connection: Upgrade
Upgrade: websocket
Origin: http://localhost:6060
Sec-WebSocket-Version: 13
User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.122 Safari/537.36
Accept-Encoding: gzip,deflate,sdch
Accept-Language: en-US,en;q=0.8,zh-CN;q=0.6,zh;q=0.4,zh-TW;q=0.2,ja;q=0.2
Sec-WebSocket-Key: 2DbWGFVjcauSVjY1+/2neQ==
Sec-WebSocket-Extensions: permessage-deflate; client_max_window_bits

```

如果服务器支持 websocket 协议，同样通过 http 头中的 Upgrade 属性来表示同意进行协议的转换，下面是一个响应 http 头的示例：  

```
HTTP/1.1 101 Switching Protocols
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Accept: w6rXmjUVLpQxK3rHk25Va9h7Y2w=

```

#### 2.2.2 基于 Websocket 协议的实时系统监控工具  

在 Go 语言中，我们可以使用`github.com/gorilla/websocket` 来实现 websocket 协议。在实际应用中，我们判断每一个请求的 header 中`Upgrade`是否包含`“websocket”`字符串，`Connection` 字段中是否包含`“Upgrade”`字符串，如果都包含的话这就是一个 websocket 请求。  

下面让我们通过一个练习来学习下 go 语言中的 websocket 的实现，在这个练习中，我将开发一个简单的基于 websocket 的服务器监控软件`sysm`，它将系统的 CPU 和内存使用率实时显示在浏览器页面上，浏览器端的绘图我将使用 Highcharts 进行绘图。关于 HighCharts 的更多信息可以参考：[HighCharts 官方文档](http://www.highcharts.com/)。  

总的来说`sysm`的工作流程如下：  

1. `sysm`程序启动，开始监听端口；  
2. `sysm`程序判断 http 请求是否是 websocket 请求，如果不是则发送页面资源，页面中包含使用 HighCharts 绘图的代码以及 websocket 连接代码；  
3. 当浏览器加载完毕`sysm`发送的静态页面后，浏览器开始执行 websocket 连接代码；  
4. `sysm`服务端检测到 http 请求头是 websocket 连接，然后开始发送 CPU、内存使用率数据；  
5. 此时浏览器页面中的 websocket 连接，收到数据，开始进行绘图工作；  

可以看到整个逻辑比较简单，针对每一个功能模块，我们可以将源代码划分为以下几个部分：  

- http_server.go  
  实现了一个功能简单的 http 服务器，该服务器针对请求是否是 websocket 请求做出相应的处理；  

- websocket.go  
  基于`github.com/gorilla/websocket`模块实现了基本的 websocket 操作；  

- template.go  
  主要是实现了对页面资源的操作，在浏览器对服务器第一次发起请求的时候，我们可以使用`Template`将页面资源发送给浏览器；  

- watcher.go  
  主要实现了系统 CPU 和内存使用率的监控；  

- sysm.go  
  封装了以上代码以方便对外使用。  

