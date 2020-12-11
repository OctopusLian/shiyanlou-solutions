#! /usr/bin/python3

import ddt
import unittest
# baseinfo包中的__init__.py文件为存放数据文件
import baseinfo


@ddt.ddt
class MyTest(unittest.TestCase):
    def setUp(self):
        print("setUp")
    def tearDown(self):
        print("tearDown")
    # 数据提取采用调用方法的方式
    @ddt.data(*baseinfo.data)
    def test01(self, testdata):
        # print("start")
        print(testdata)
        # print("end")


if __name__ == '__main__':
    unittest.main()