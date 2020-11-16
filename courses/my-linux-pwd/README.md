# 实现linux操作系统下的pwd命令  

## 目标  

- linux pwd 命令的使用  
- linux文件系统中文件及目录的实现方式  
- linux文件及目录系统调用接口的使用  

## 实现思路  

1. 通过特殊的文件名“.”获取当前目录的inode-number(假设当前目录为a)  
2. 通过特殊的文件名“..”获取当前目录的父级目录的inode-number  
3. 判断当前目录和上级目录的inode-number是否一样  
4. 如果两个inode-number一样说明到达根目录，输出完整路径，退出程序  
5. 如果两个inode-number不一样，切换至 父级目录，根据步骤1获取的inode-number，在父级目录中搜索对应的文件名并记录下来，然后重新回到步骤1.  

## 编译运行  

编译  
```
$ gcc -o mypwd mypwd.c
```

运行  
```
$ ./mypwd
```
