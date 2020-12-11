#! /usr/bin/python3

from selenium import webdriver
from time import sleep

driver = webdriver.Firefox()
driver.get("http://bbs.51testing.com/forum.php")
sleep(3)

# 页面下拉指定高度
js = 'document.documentElement.scrollTop=800;'
driver.execute_script(js)
