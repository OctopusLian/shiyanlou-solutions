#! /usr/bin/python3

from selenium import webdriver
from time import sleep


driver = webdriver.Firefox()

# 进入51testing网站
driver.get("http://bbs.51testing.com/forum.php")
sleep(3)

# 用id定位账号输入框并输入账号
driver.find_element_by_id("ls_username").send_keys("您的用户名")

# 用id定位密码输入框并输入密码
driver.find_element_by_id("ls_password").send_keys("密码")

# 定位“登录”按钮并获取登录按钮的文本
txt = driver.find_element_by_xpath('//*[@id="lsform"]/div/div[1]/table/tbody/tr[2]/td[3]/button').text

# 打印获取的文本
print(txt)

# 定位“登录”按钮并获取登录按钮的type属性值
type = driver.find_element_by_xpath('//*[@id="lsform"]/div/div[1]/table/tbody/tr[2]/td[3]/button').get_attribute("type")

# 打印type属性值
print(type)

# 定位“登录”按钮并进行点击操作
driver.find_element_by_xpath('//*[@id="lsform"]/div/div[1]/table/tbody/tr[2]/td[3]/button').click()