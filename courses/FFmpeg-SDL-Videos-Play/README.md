# ffmpeg结合sdl

## 知识点  

- ffmpeg 库介绍  
- 相关结构介绍  
- 动画屏幕捕捉  
- 格式转换  

## 编译运行  
编译  
```
gcc -Wall -ggdb main.c  -I/usr/local/include -L/usr/local/lib -lavformat -lavcodec -lva-x11 -lva -lxcb-shm -lxcb-xfixes -lxcb-render -lxcb-shape -lxcb -lX11 -lasound -lz -lswresample -lswscale -lavutil -lm `sdl-config --cflags --libs` -pthread -o main.out
```

运行  
```
./main.out mp4文件的绝对路径
```