# 实验介绍  

如果你记得小时候在小霸王上玩过一个叫做 TT 打字游戏的话，那说明你已经老了。不知道的话，那么恭喜你，现在有机会玩一下这个有点老，但是却有趣又有用的简单小游戏了，不对，应该是自己亲手写一个来玩。这个打字游戏要实现的功能很简单，就是在屏幕上显现一些字符或单词，然后你输入相应的字符或单词，如果输入正确的话，屏幕上的单词就会消失，并会出现一个新的词，以此来使计算机初学者熟悉键盘操作和练习打字。小伙伴们应该在之前都学习过 shell 脚本了，想不想要展现一下你的实力水平呢，没学懂？学了太久很多东西不记得了？没关系，要是大家都会了，还要我们干嘛呢 (-: 这次我们就是要用 shell 脚本来实现这个游戏，顺便让大家练练手，复习复习之前学习的东西。废话少说，让我们开始吧。  

## 实验知识点  

shell 程序设计与编写  
shell 数组字符串操作  
shell 基本语法  
linux 常见命令用法  

## 实验环境  
本实验环境采用带桌面的 Ubuntu Linux 环境，实验中会用到桌面上的程序：  

- Xfce 终端：Linux 命令行终端，打开后会进入 Bash 环境，可以使用 Linux 命令。  
- gvim：非常好用的 Vim 编辑器，最简单的用法可以参考课程 Vim 编辑器。  
- 其他编辑器：如果 Vim 不熟悉可以使用 gedit 或 brackets。  

## 代码获取  

```
$ git clone https://github.com/shiyanlou/shell_typing_game

说明：
game.sh 是主程序，输入 bash game.sh 执行
word 是单词文件
word.txt 是未处理单词文件
```

## 课后习题  

### 修改游戏背景颜色  

```sh
144 echo -e "\033[44m\033["$i";"$j"H "   

它的参数含义是  
echo -e  "\033[前景颜色;背景颜色m, \033[行；列 H"  
```

### 修改字母下落的速度  

```sh
327  if read -n 1 -t 0.5 tmp
```

-t参数，便是指定输入超时时间，即当用户在超时时间内未输入或输入未结束的话，会立即结束读取，所以在整个代码的循环里，只要缩短这个时间，就能加快字符下落的速度，把它改成0.1试试呢