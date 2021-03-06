#! /usr/bin/python3

from selenium import webdriver
from time import sleep


def get_results(filename):
    driver = webdriver.Firefox()
    driver.maximize_window()
    result_url = "file://%s" % filename
    driver.get(result_url)
    sleep(3)
    res = driver.find_element_by_xpath("/html/body/div[1]/p[4]").text
    result = res.split(':')
    driver.quit()
    return result[-1]
