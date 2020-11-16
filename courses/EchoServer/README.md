# 实现 echo服务器/客户端  

## 服务端的实现思路及步骤:  
1. 创建一个套接字对象, 指定其IP以及端口.  
2. 开始监听套接字指定的端口.  
3. 如有新的客户端连接请求, 则建立一个goroutine, 在goroutine中, 读取客户端消息, 并转发回去, 直到客户端断开连接  
4. 主进程继续监听端口.  

## 客户端的代码实现步骤:  
1. 创建一个套接字对象, ip与端口指定到上面我们实现的服务器的ip与端口上.  
2. 使用创建好的套接字对象连接服务器.  
3. 连接成功后, 开启一个goroutine, 在这个goroutine内, 定时的向服务器发送消息, 并接受服务器的返回消息, 直到错误发生或断开连接.  

## 第一次运行  
服务端  
```
./server 
A client connected : 127.0.0.1:48382
time

2019-09-05 22:58:45.947274321 +0800 CST m=+3.424810972

2019-09-05 22:58:46.948315309 +0800 CST m=+4.425851875

2019-09-05 22:58:47.949076749 +0800 CST m=+5.426613279

2019-09-05 22:58:48.949454356 +0800 CST m=+6.426990874

2019-09-05 22:58:49.949932746 +0800 CST m=+7.427469253

2019-09-05 22:58:50.950439533 +0800 CST m=+8.427976107

2019-09-05 22:58:51.95082671 +0800 CST m=+9.428363237
```

客户端  
```
./client 
connected!
2019-09-05 22:58:45.947274321 +0800 CST m=+3.424810972

2019-09-05 22:58:46.948315309 +0800 CST m=+4.425851875

2019-09-05 22:58:47.949076749 +0800 CST m=+5.426613279

2019-09-05 22:58:48.949454356 +0800 CST m=+6.426990874

2019-09-05 22:58:49.949932746 +0800 CST m=+7.427469253

2019-09-05 22:58:50.950439533 +0800 CST m=+8.427976107

2019-09-05 22:58:51.95082671 +0800 CST m=+9.428363237

2019-09-05 22:58:52.951572467 +0800 CST m=+10.429108959
```

## echo服务器/客户端改造成一个文本信息的聊天室  
服务端  
```
./server 
A client connected : 127.0.0.1:48426
Hi

Hello

thanks
```

客户端  
```
./client 
connected!
Hi
127.0.0.1:48426:Hi

Hello
127.0.0.1:48426:Hello

thanks
127.0.0.1:48426:thanks
```