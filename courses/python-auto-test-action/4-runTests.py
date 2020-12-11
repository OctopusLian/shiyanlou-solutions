# coding: utf-8

import unittest

discover = unittest.defaultTestLoader.discover("/home/shiyanlou/Desktop/MySelenium/testCases/", pattern="test*.py")
print(discover)
runner = unittest.TextTestRunner()
runner.run(discover)
