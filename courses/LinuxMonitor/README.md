## Shell 脚本实现 Linux 系统监控  

实现一个包含各种不同参数的 Shell 脚本，用来获取和监控 Linux 系统信息，并将该脚本加入系统环境中。  
实现过程中学习和实践 Shell 脚本编程及 Linux 基本信息和资源使用率获取。  

### 知识点  
- `Bash`脚本编程  
- 如何获取`Linux`系统信息  
- 如何实时获取`Linux`资源使用率  


### 运行结果  
```
$ bash monitor.sh

Operating System Type :  GNU/Linux
OS Version :  Linux Debian buster/sid ( 4.18.0-25-generic x86_64)
Architecture :  x86_64
Kernel Release :  4.18.0-25-generic
Hostname :  zoctopus
Internal IP :  192.168.1.6 172.17.0.1 2409:8a62:375:d840:15b0:bc83:bf7e:32ab 2409:8a62:375:d840:ad69:558f:aac8:7cc4
External IP :  112.45.96.57
Name Servers :  This internal configured Run currently Third symlink replace See operation 127.0.0.53 edns0
Logged In users : 
zoctopus :0           2019-07-09 21:08 (:0)
Ram Usages : 
              total        used        free      shared  buff/cache   available
Mem:           3.7G        1.6G        233M        244M        1.9G        1.7G
Swap Usages : 
              total        used        free      shared  buff/cache   available
Swap:          2.0G        1.0G        989M
Disk Usages : 
Filesystem      Size  Used Avail Use% Mounted on
/dev/sda2       228G   22G  195G  10% /
/dev/sda1       511M  6.1M  505M   2% /boot/efi
/dev/sdb4       7.5G  2.4G  5.1G  32% /media/zoctopus/Ubuntu 18.0
Load Average :  loadaverage:0.52,
System Uptime Days/(HH:MM) :  6 days

```

### 注意  

本实验源自[TECMINT_MONITOR](https://github.com/atarallo/TECMINT_MONITOR)

