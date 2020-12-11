#! /usr/bin/python3

from selenium import webdriver
from time import sleep


driver = webdriver.Firefox()

# 浏览器进入百度网站
driver.get("https://www.baidu.com")

# 设置浏览器宽800，高400
driver.set_window_size(800, 400)

# 等待3秒
sleep(3)

# 最大化窗口
driver.maximize_window()
# 进入另一个网站
driver.get("https://www.lanqiao.cn/")
sleep(3)

# 后退到上一个页面--百度网站
driver.back()

sleep(3)

# 前进到下一个页面--实验楼网站
driver.forward()

sleep(3)

# 退出浏览器
driver.quit()