#! /usr/bin/python3

from selenium import webdriver
from time import sleep


driver = webdriver.Firefox()
driver.get("https://mail.163.com/")
sleep(3)
frame = driver.find_element_by_xpath('//*[@id="loginDiv"]/iframe')
# 切换到iframe
driver.switch_to.frame(frame)
account = driver.find_element_by_xpath('//*[@id="account-box"]/div[2]/input').get_attribute("data-placeholder")
print(account)
assert account == '邮箱帐号或手机号码'
driver.find_element_by_xpath('//*[@id="account-box"]/div[2]/input').send_keys("您的账号")
passwd = driver.find_element_by_xpath("//*[@id='login-form']/div/div[3]/div[2]/input[2]").get_attribute("data-placeholder")
print(passwd)
assert passwd == '输入密码'
driver.find_element_by_xpath("//*[@id='login-form']/div/div[3]/div[2]/input[2]").send_keys("您的密码")
driver.find_element_by_xpath('//*[@id="dologin"]').click()
# 切换出iframe
driver.switch_to.default_content()
sleep(3)
name = driver.find_element_by_xpath("//*[@id='dvContainer']/div/div/div[2]/div/div[2]/span/span").text
print(name)
assert name == '您的用户名'