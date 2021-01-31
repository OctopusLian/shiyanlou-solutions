## Linux Sniffer  
Linux网络嗅探器：拦截通过网络接口流入和流出的数据的程序。  
#### 运行  
```
$ make

$ sudo ./network_sniffer
```
#### 文件目录  
```
make前：
.
├── main.c
├── Makefile
├── README.md
├── show_data.c
├── sniffer.h
├── tools.c
└── tools.h

0 directories, 7 files


make后：
.
├── main.c
├── main.o
├── Makefile
├── network_sniffer
├── README.md
├── show_data.c
├── show_data.o
├── sniffer.h
├── tools.c
├── tools.h
└── tools.o

0 directories, 11 files
```