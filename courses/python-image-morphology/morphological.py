#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 13:40:28 2018

@author: Berry
"""

import numpy as np
from PIL import Image

class ImageMorphological():
    def __init__(self, threshold = 120, image_path= './pics/test.jpg'):
        '''Init.
        
        Args: 
            image_path: path to the image needed to process
        '''
        self.image = np.asarray(Image.open(image_path))
        self.wb = self.gray2wb(threshold=threshold)
        
    
    def gray2wb(self, threshold = 120):
        wb = (self.image >= threshold) * 255
        wb = wb.astype(np.uint8)
        return wb
    
    def pad(self, img, row = 1, col = 1, mode = 'constant'):
        '''Padding an image.
        
        Args:
            img: a numpy array
            row: padding size for rows (an odd number)
            clo: padding size for columns (an odd number)
            
        Returns:
            image_pad: a new image array (zero padding)
        '''
        return np.pad(img, ((row, row), (col, col)), mode)
        
    def is_interset(self, kernel, img_slice):
        '''To judge if the kernel has intersection with the image slice.
        
        Args:
            kernel: se (a numpy array)
            img_slice: a slice of image (array)
            
        Returns:
            true or false
        '''
        for i, row in enumerate(kernel):
            for j, value in enumerate(row):
                if value == 255 and img_slice[i, j] == 255:
                    return True
                if value == 255 and img_slice[i, j] == 0:
                    continue
        return False
    
    def is_cover(self, kernel, img_slice):
        '''To judge if the kernel is covered by the image slice.
        
        Args:
            kernel: se (a numpy array)
            img_slice: a slice of image (array)
            
        Returns:
            true or false
        '''
        for i, row in enumerate(kernel):
            for j, value in enumerate(row):
                if value == 255 and img_slice[i, j] == 255:
                    continue
                if value == 255 and img_slice[i, j] == 0:
                    return False
        return True
    
    def erosion_bin(self, img, kernel, is_save = True, save_path = './pics/erosion.png'):
        '''Erosion Fucntion.
        
        Args:
            img: a numpy array of size (W, H)
            kernel: a numpy array of size (w, h) and w, h must be odd numbers
            is_save: True to save, False not save
            save_path: save path
        '''
        w, h = kernel.shape
        img_pad = self.pad(img, (w-1)//2, (h-1)//2)
        tmp = np.zeros(img.shape)
        for r, row in enumerate(img):
            for c, value in enumerate(row):
                covered = self.is_cover(kernel, img_pad[r:r+w, c:c+h])
                if covered:
                    tmp[r, c] = 255
                else:
                    tmp[r, c] = 0
        if is_save:
            self.imsave(tmp, save_path)
        else:
            return tmp
        
    def dilation_bin(self, img, kernel, is_save = True, save_path = './pics/dilation.png'):
        '''Dilation. I use the center as the origin location.
        
        Args:
            img: a numpy array of size (W, H)
            kernel: a numpy array of size (w, h) and w, h must be odd numbers
            is_save: True to save, False not save
            save_path: save path
        '''
        w, h = kernel.shape
        img_pad = self.pad(img, (w-1)//2, (h-1)//2)
        tmp = np.zeros(img.shape)
        for r, row in enumerate(img):
            for c, value in enumerate(row):
                intersected = self.is_interset(kernel, img_pad[r:r+w, c:c+h])
                if intersected:
                    tmp[r, c] = 255
                else:
                    tmp[r, c] = 0
                    
        if is_save:
            self.imsave(tmp, save_path)
        else:
            return tmp
    
    def local_min(self, mask, img_slice):
        '''Find the local minimum value of img_slice.
        
        Args:
            mask: a mask 
            img_slice: image slice
        '''
        index = np.sum(mask).astype(np.int64)
        tmp = img_slice * mask
        return np.sort(tmp.reshape((1,-1)))[0][-index]
    
    def local_max(self, mask, img_slice):
        '''Find the local maximum value of img_slice.
        
        Args:
            mask: a mask 
            img_slice: image slice
        '''
        tmp = img_slice * mask
        return np.max(tmp)
    
    def erosion_gray(self, img, kernel, is_save = True, save_path = './pics/erosion_gray.png'):
        '''Erosion Fucntion.
        
        Args:
            img: a numpy array of size (W, H)
            kernel: a numpy array of size (w, h) and w, h must be odd numbers
            is_save: True to save, False not save
            save_path: save path
        '''
        w, h = kernel.shape
        img_pad = self.pad(img, (w-1)//2, (h-1)//2, mode='edge')
        tmp = np.zeros(img.shape)
        
        for r, row in enumerate(img):
            for c, value in enumerate(row):
                tmp[r, c] = self.local_min(kernel, img_pad[r:r+w, c:c+h])
        if is_save:
            self.imsave(tmp, save_path)
        else:
            return tmp
    
    def dilation_gray(self, img, kernel, is_save = True, save_path = './pics/dilation_gray.png'):
        '''Dilation. I use the center as the origin location.
        
        Args:
            img: a numpy array of size (W, H)
            kernel: a numpy array of size (w, h) and w, h must be odd numbers
            is_save: True to save, False not save
            save_path: save path
        '''
        w, h = kernel.shape
        img_pad = self.pad(img, (w-1)//2, (h-1)//2, mode='edge')
        tmp = np.zeros(img.shape)
        
        for r, row in enumerate(img):
            for c, value in enumerate(row):
                tmp[r, c] = self.local_max(kernel, img_pad[r:r+w, c:c+h])
        if is_save:
            self.imsave(tmp, save_path)
        else:
            return tmp
    
    def imopen(self, img, se_erosion, se_dilation, save_path = './pics/open.png', mode = "bin"):
        '''First erosion and then dilation.
        
        Args:
            img: a numpy array of size (W, H)
            se_erosion: erosion kernel
            se_dilation: dilation kernel
            save_path: save path
            mode: "bin" or "gray"
        '''
        tmp = np.zeros(img.shape)
        if mode == 'bin':
            tmp = self.erosion_bin(img, se_erosion, False)
            tmp = self.dilation_bin(tmp, se_dilation, False)
        elif mode == 'gray':
            tmp = self.erosion_gray(img, se_erosion, False)
            tmp = self.dilation_gray(tmp, se_dilation, False)
        self.imsave(tmp, save_path)
        
    def imclose(self, img, se_erosion, se_dilation, save_path = './pics/close.png', mode = "bin"):
        '''First dialtion and then erosion.
        
        Args:
            img: a numpy array of size (W, H)
            se_erosion: erosion kernel
            se_dilation: dilation kernel
            save_path: save path
            mode: "bin" or "gray"
        '''
        tmp = np.zeros(img.shape)
        if mode == 'bin':
            tmp = self.dilation_bin(img, se_dilation, False)
            tmp = self.erosion_bin(tmp, se_erosion, False)
        elif mode == 'gray':
            tmp = self.dilation_gray(img, se_dilation, False)
            tmp = self.erosion_gray(tmp, se_erosion, False)
        self.imsave(tmp, save_path)
    
    def imsave(self, imarray, path = './pics/wb.png'):
        tmp = Image.fromarray(np.uint8(imarray))
        tmp.save(path, 'png')

if __name__ == '__main__':
    # load image
    img = ImageMorphological(threshold=120, image_path='./pics/test.jpg')
    image = img.image
    print(type(image))
    print(image.shape)
    
    # gray2wb
    img.imsave(img.wb, './pics/wb.png')
    
    #Binary
    # Defint 4 kernels
    kernel1 = np.array([[0, 255, 0],
                        [255, 255, 255],
                        [0, 255, 0]])
    kernel2 = np.ones((3,3))*255
    kernel3 = np.array([[0,0,255,0,0],
                        [0,255,255,255,0],
                        [255,255,255,255,255],
                        [0,255,255,255,0],
                        [0,0,255,0,0]])
    kernel4 = np.ones((5,5))*255
    
    # binary kernels erosion
    img.erosion_bin(img.wb, kernel1, save_path='./pics/erosion_bin_k1.png')
    img.erosion_bin(img.wb, kernel2, save_path='./pics/erosion_bin_k2.png')
    img.erosion_bin(img.wb, kernel3, save_path='./pics/erosion_bin_k3.png')
    img.erosion_bin(img.wb, kernel4, save_path='./pics/erosion_bin_k4.png')
    
    # binary kernels dilation
    img.dilation_bin(img.wb, kernel1, save_path='./pics/dilation_bin_k1.png')
    img.dilation_bin(img.wb, kernel2, save_path='./pics/dilation_bin_k2.png')
    img.dilation_bin(img.wb, kernel3, save_path='./pics/dilation_bin_k3.png')
    img.dilation_bin(img.wb, kernel4, save_path='./pics/dilation_bin_k4.png')
    
    # binary kernels open
    img.imopen(img.wb, kernel1, kernel1, save_path='./pics/open_binary_k11.png')
    img.imopen(img.wb, kernel2, kernel2, save_path='./pics/open_binary_k22.png')
    img.imopen(img.wb, kernel3, kernel3, save_path='./pics/open_binary_k33.png')
    img.imopen(img.wb, kernel4, kernel4, save_path='./pics/open_binary_k44.png')
    
    # binary kernels close
    img.imclose(img.wb, kernel1, kernel1, save_path='./pics/close_binary_k11.png')
    img.imclose(img.wb, kernel2, kernel2, save_path='./pics/close_binary_k22.png')
    img.imclose(img.wb, kernel3, kernel3, save_path='./pics/close_binary_k33.png')
    img.imclose(img.wb, kernel4, kernel4, save_path='./pics/close_binary_k44.png')
    
    # GRAY
    mask1 = np.array([[0,1,0],
                      [1,1,1],
                      [0,1,0]])
    mask2 = np.ones((3,3))
    mask3 = np.array([[0,0,1,0,0],
                      [0,1,1,1,0],
                      [1,1,1,1,1],
                      [0,1,1,1,0],
                      [0,0,1,0,0]])
    mask4 = np.ones((5,5))
    
    #gray kernels erosion
    img.erosion_gray(img.image, mask1, save_path='./pics/erosion_gray_k1.png')
    img.erosion_gray(img.image, mask2, save_path='./pics/erosion_gray_k2.png')
    img.erosion_gray(img.image, mask3, save_path='./pics/erosion_gray_k3.png')
    img.erosion_gray(img.image, mask4, save_path='./pics/erosion_gray_k4.png')
    
    # gray kernels dilation
    img.dilation_gray(img.image, mask1, save_path='./pics/dilation_gray_k1.png')
    img.dilation_gray(img.image, mask2, save_path='./pics/dilation_gray_k2.png')
    img.dilation_gray(img.image, mask3, save_path='./pics/dilation_gray_k3.png')
    img.dilation_gray(img.image, mask4, save_path='./pics/dilation_gray_k4.png')
    
    # gray kernels open    
    img.imopen(img.image, mask1, mask1, save_path='./pics/open_gray_k11.png', mode='gray')
    img.imopen(img.image, mask2, mask2, save_path='./pics/open_gray_k22.png', mode='gray')
    img.imopen(img.image, mask3, mask3, save_path='./pics/open_gray_k33.png', mode='gray')
    img.imopen(img.image, mask4, mask4, save_path='./pics/open_gray_k44.png', mode='gray')
    
    #gray kernels close
    img.imclose(img.image, mask1, mask1, save_path='./pics/close_gray_k11.png', mode='gray')
    img.imclose(img.image, mask2, mask2, save_path='./pics/close_gray_k22.png', mode='gray')
    img.imclose(img.image, mask3, mask3, save_path='./pics/close_gray_k33.png', mode='gray')
    img.imclose(img.image, mask4, mask4, save_path='./pics/close_gray_k44.png', mode='gray')