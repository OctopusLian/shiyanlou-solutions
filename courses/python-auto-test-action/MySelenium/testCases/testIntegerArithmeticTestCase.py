# coding: utf-8

import unittest

class testIntegerArithmeticTestCase(unittest.TestCase):
    def testAdd(self):  # test method names begin with 'test'
        self.assertEqual((1 + 2), 3)
        self.assertEqual(0 + 1, 1)
        print("testAdd passed")
    def testMultiply(self):
        self.assertEqual((0 * 10), 0)
        self.assertEqual((5 * 8), 40)
        print("testMultiply passed")

if __name__ == '__main__':
    unittest.main()
