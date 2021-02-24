<?php 

$sock = socket_create(AF_INET, SOCK_STREAM, SOL_TCP);

socket_bind($sock, "127.0.0.1", 8080);

socket_listen($sock);

for ( ; ; ) {
    $conn = socket_accept($sock);

    $write_buffer = "HTTP/1.0 200 OK\r\nServer: my_server\r\nContent-Type: text/html; charset=utf-8\r\n\r\nhello!world";

    socket_write($conn, $write_buffer);

    socket_close($conn);
}
