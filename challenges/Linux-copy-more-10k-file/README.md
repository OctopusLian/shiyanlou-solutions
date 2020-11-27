## 介绍  

将实验楼实验环境中的 /etc 目录下的所有大于 10K 的文件拷贝到 /tmp 目录，需要保持目录结构。  

例如 /etc/apt/trusted.gpg 文件大小为 14K，则会被拷贝到 /tmp/etc/apt/trusted.gpg 路径位置。  

注意 /etc 目录下的子文件夹中也有很多文件超过 10K，需要拷贝。  

拷贝完成后点击 提交结果。  

## 目标  

1，/etc 目录下所有大于 10K 的文件（不论 shiyanlou 用户是否对该文件具有访问权限）都被拷贝到 /tmp/etc 目录下。  
２，拷贝完成后 /tmp/etc 目录中只包含大于 10K 的文件。文件的路径需要保持目录结构。  
３，请不要使用软链接等方式，需要将文件完整的拷贝过去。  

[链接](https://www.lanqiao.cn/challenges/2826/)  