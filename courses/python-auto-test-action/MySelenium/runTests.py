#! /usr/bin/python3


from HTMLTestRunner import HTMLTestRunner
from modules import sendEmail
from modules import getTestResult
from modules import dir_path
import unittest


if __name__ == '__main__':

    # 测试用例路径
    test_dir = dir_path.dir_path() + '/testCases'
    # 测试报告存放路径
    report_dir = dir_path.dir_path() + '/report'

    filename = report_dir + '/report.html'
    fp = open(filename, 'wb')
    runner = HTMLTestRunner(stream=fp, title='自动化测试', description='用例执行结果')
    discover = unittest.defaultTestLoader.discover(test_dir, pattern='test*.py')
    runner.run(discover)
    fp.close()
    result = getTestResult.get_results(filename)
    mail = sendEmail.send_Mail(filename, result)
    if mail:
        print(u'邮件发送成功！')
    else:
        print(u'邮件发送失败！')
