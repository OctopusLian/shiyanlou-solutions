# Socket介绍  

### socket_create  

TODO ： 创建一个新的 socket 资源 函数原型: resource socket_create ( int $domain , int $type , int $protocol ) 它包含三个参数，分别如下：  
- domain：AF_INET、AF_INET6、AF_UNIX，AF的释义就 address family，地址族的意思，我们常用的有 ipv4、ipv6  
- type: SOCK_STREAM、SOCK_DGRAM等，最常用的就是SOCK_STREAM，基于字节流的SOCKET类型，也是TCP协议使用的类型  
- protocol: SOL_TCP、SOL_UDP 这个就是具体使用的传输协议，一般可靠的传输我们选择 TCP，游戏数据传输我们一般选用 UDP 协议  

### socket_bind  

TODO ： 将创建的 socket 资源绑定到具体的 ip 地址和端口 函数原型: bool socket_bind ( resource $socket , string $address [, int $port = 0 ] )  
它包含三个参数，分别如下：  
- socket: 使用socket_create创建的 socket 资源，可以认为是 socket 对应的 id  
- address: ip 地址  
- port: 监听的端口号，WEB 服务器默认80端口  

### socket_listen  

TODO ： 在具体的地址下监听 socket 资源的收发操作 函数原型: bool socket_listen ( resource $socket [, int $backlog = 0 ] )  
它包含两个个参数，分别如下：  
- socket: 使用socket_create创建的socket资源  
- backlog: 等待处理连接队列的最大长度  

### socket_accept  

TODO ： 接收一个新的 socket 资源 函数原型: resource socket_accept ( resource $socket )  
- socket: 使用socket_create创建的socket资源  

### socket_write  

TODO ： 将指定的数据发送到 对应的 socket 管道 函数原型: int socket_write ( resource $socket , string $buffer [, int $length ] )  
- socket: 使用socket_create创建的socket资源  
- buffer: 写入到socket资源中的数据  
- length: 控制写入到socket资源中的buffer的长度，如果长度大于buffer的容量，则取buffer的容量  

### socket_read  

TODO ： 获取传送的数据 函数原型: int socket_read ( resource $socket , int $length )  
- socket: 使用socket_create创建的socket资源  
- length: socket资源中的buffer的长度  

### socket_close  

TODO ： 关闭 socket 资源 函数原型: void socket_close ( resource $socket )  
- socket: socket_accept或者socket_create产生的资源，不能用于stream资源的关闭  

### stream_socket_server  

由于创建一个SOCKET的流程总是 socket、bind、listen，所以PHP提供了一个非常方便的函数一次性创建、绑定端口、监听端口  

函数原型: resource stream_socket_server ( string $local_socket [, int &$errno [, string &$errstr [, int $flags = STREAM_SERVER_BIND | STREAM_SERVER_LISTEN [, resource $context ]]]] )  

- local_socket: 协议名://地址:端口号  
- errno: 错误码  
- errstr: 错误信息  
- flags: 只使用该函数的部分功能  
- context: 使用stream_context_create函数创建的资源流上下文  

# 运行  

执行命令  
```
sudo service php7.2-fpm start;
php server.php
```

运行成功后，打开浏览器，输入http://127.0.0.1:8080，回车看到结果hello!world，同时，我们也可以使用集成函数stream_socket_server去改写成：   

```php
<?php 

$sock = stream_socket_server("tcp://127.0.0.1:8080", $errno, $errstr);

for ( ; ; ) {
    $conn = stream_socket_accept($sock);

    $write_buffer = "HTTP/1.0 200 OK\r\nServer: my_server\r\nContent-Type: text/html; charset=utf-8\r\n\r\nhello!world";

    fwrite($conn, $write_buffer);

    fclose($conn);
}
```

## 注意  

这里不能使用socket_accept，因为stream_socket_server和socket_create创建的不是同一种资源，stream_socket_server创建的是stream资源，这也是为什么可以用fwrite、fread、fclose操作该资源的原因. 而socket_create创建的是socket资源，而不是stream资源，所以socket_create创建的资源只能用socket_write、socket_read、socket_close来操作.
