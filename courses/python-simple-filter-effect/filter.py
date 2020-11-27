"""
filter can add filter to the photo.

Usage: filter [options] <curves> <filepath>

Options:
    -h --help     show the help message
    -v --version  show the version information
"""

from struct import unpack
from scipy import interpolate

from PIL import Image
import numpy as np
import scipy

import sys
from docopt import docopt


__version__ = '1.0'

# 滤镜类
class Filter:
    def __init__(self, acv_file_path, name):
        self.name = name
        # 打开文件
        with open(acv_file_path, 'rb') as acv_file:
            # 获取曲线点坐标集
            self.curves = self._read_curves(acv_file)
        # 由曲线的点坐标集计算得曲线对应的多项式
        self.polynomials = self._find_coefficients()
    # 从 acv 文件中获取 curves
    # https://github.com/vbalnt/filterizer
    def _read_curves(self, acv_file):
        # 首先读取前 4 个字节
         # nr_curves 是文件中曲线的数目
        _, nr_curves = unpack('!hh', acv_file.read(4))
        curves = []
        # 遍历所有曲线，获取各个曲线的点集
        for i in range(nr_curves):
            curve = []
            # 获取曲线点的个数
            num_curve_points, = unpack('!h', acv_file.read(2))
            for j in range(num_curve_points):
                y, x = unpack('!hh', acv_file.read(4))
                curve.append((x,y))
            curves.append(curve)
        return curves
    # 获取各通道多项式系数
    # 返回多项式
    def _find_coefficients(self):
        polynomials = []
        for curve in self.curves:
            xdata = [x[0] for x in curve]
            ydata = [x[1] for x in curve]
            # 通过 lagrange 插值获取多项式
            # 多项式的类型为 numpy.poly1d
            p = interpolate.lagrange(xdata, ydata)
            polynomials.append(p)
        return polynomials
    # 获取红色通道的多项式
    def get_r(self):
        return self.polynomials[1]
    # 获取绿色通道的多项式
    def get_g(self):
        return self.polynomials[2]
    # 获取蓝色通道的多项式
    def get_b(self):
        return self.polynomials[3]
    # 获取混合通道的多项式
    def get_c(self):
        return self.polynomials[0]
# 滤镜管理类
class FilterManager:
    def __init__(self):
        self.filters = {}
    # 添加滤镜
    def add_filter(self, filter_obj):
        self.filters[filter_obj.name] = filter_obj
    # 应用滤镜
    def apply_filter(self, filter_name, image_array):
        if image_array.ndim < 3:
            raise Exception('Photos must be in color, meaning at least 3 channels')
        else:
            def interpolate(i_arr, f_arr, p, p_c):
                p_arr = p_c(f_arr)
                return p_arr 
        # 获取滤镜
        image_filter = self.filters[filter_name]
        # 获取图像宽高以及通道信息
        width, height, channels = image_array.shape
        # 新建图像数组，用于存储处理之后的图像数据
        filter_array = np.zeros((width, height, 3), dtype=float)
        # 分别获取红、绿、蓝和混合通道的多项式
        p_r = image_filter.get_r()
        p_g = image_filter.get_g()
        p_b = image_filter.get_b()
        p_c = image_filter.get_c()
        # 对图像中每个相应通道进行处理
        filter_array[:,:,0] = p_r(image_array[:,:,0])
        filter_array[:,:,1] = p_g(image_array[:,:,1])
        filter_array[:,:,2] = p_b(image_array[:,:,2])
        # 保证数据位于 0~255 之间，防止数据溢出
        filter_array = filter_array.clip(0,255)
        filter_array = p_c(filter_array)
        # numpy.ceil 将向上取整
        filter_array = np.ceil(filter_array).clip(0,255)
         # 将图像数据格式转换为无符号8位整型
        return filter_array.astype(np.uint8)
# 分割文件路径名，获得文件名
# 例如： str = /usr/home/123.jpg 函数返回文件名 123
def get_name(str):
    return str.split('/')[-1].split('.')[0]

def main():
    # 构建语法解析器
    args = docopt(__doc__, version=__version__)
    # 创建一个滤镜类 Filter 的实例
    img_filter = Filter(args['<curves>'], 'crgb')
    # 程序读取指定图片
    im = Image.open(args['<filepath>'])
    # 转换数据类型，便于后续处理
    image_array = np.array(im)
    # 创建滤镜管理类 FilterManager 对象实例
    filter_manager = FilterManager()
    # 加载指定滤镜
    filter_manager.add_filter(img_filter)
    # 使用滤镜处理图像数据数组
    filter_array = filter_manager.apply_filter('crgb', image_array)
    # 转换类型
    im = Image.fromarray(filter_array)
    # 定义输出图像名
    output_name = '%s_%s.png' % (get_name(args['<filepath>']), get_name(args['<curves>']))
    # 保存图像
    im.save(output_name)

if __name__ == '__main__':
  main()