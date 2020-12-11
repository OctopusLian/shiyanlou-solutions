#! /usr/bin/python3

from os import path

def dir_path():
    file_path = path.dirname(__file__)
    # 返回父级目录
    return path.dirname(file_path)