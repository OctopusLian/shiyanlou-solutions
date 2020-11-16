# 200行Python代码实现2048  

## 知识点  

1. Python基本知识  
2. 状态机的概念  

## 主逻辑  

### 用户行为  

所有的有效输入都可以转换为"上，下，左，右，游戏重置，退出"这六种行为，用 `actions` 表示

    actions = ['Up', 'Left', 'Down', 'Right', 'Restart', 'Exit']

有效输入键是最常见的 W（上），A（左），S（下），D（右），R（重置），Q（退出），这里要考虑到大写键开启的情况，获得有效键值列表：

    letter_codes = [ord(ch) for ch in 'WASDRQwasdrq']

将输入与行为进行关联：

    actions_dict = dict(zip(letter_codes, actions * 2))

###状态机

处理游戏主逻辑的时候我们会用到一种十分常用的技术：状态机，或者更准确的说是有限状态机（FSM）

你会发现 2048 游戏很容易就能分解成几种状态的转换。


![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid8834labid1172timestamp1468333673772.png/wm)

``state`` 存储当前状态， ``state_actions`` 这个词典变量作为状态转换的规则，它的 key 是状态，value 是返回下一个状态的函数：

```
+ Init: init()
    + Game
+ Game: game()
    + Game
    + Win
    + GameOver
    + Exit
+ Win: lambda: not_game('Win')
    + Init
    + Exit
+ Gameover: lambda: not_game('Gameover')
    + Init
    + Exit
+ Exit: 
    退出循环
```

状态机会不断循环，直到达到 Exit 终结状态结束程序。  