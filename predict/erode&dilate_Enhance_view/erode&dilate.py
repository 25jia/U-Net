import cv2 
import numpy as np  
import glob
import os

for files in glob.glob('out1'+"/*_bi.png"):
    filepath,filename = os.path.split(files)
    img=cv2.imread(files)  
    #GrayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,image=cv2.threshold(img,153,255,cv2.THRESH_BINARY)  #0,1二值化
    print(image.shape)
#参数（核结构，核大小）
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 椭圆形结构
    #kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 3)) # 十字结构
    #kernel3= cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))# 十字结构
#参数（开、闭操作，卷积核）
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel1)#开操作除白噪声
    #cv2.imwrite('image1.png', image)

    #image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel2)#闭操作除黑噪声
    #cv2.imwrite('image2.png', image)

    #image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel3)#开操作修路
    image = image[:, :, 1]
    print(image.shape)
    cv2.imwrite(filename, image)


#腐蚀函数
#cv2.erode(src, kernel, dst=None, anchor=None, iterations=None, borderType=None, borderValue=None)
#膨胀函数
#cv2.dilate(src, kernel, dst=None, anchor=None, iterations=None, borderType=None, borderValue=None)
#获取不同形状的结构元素，返回指定形状和尺寸的结构元素
#cv2.getStructuringElement(shape, ksize, anchor=None)
#参数shape:表示内核的形状
#矩形：MORPH_RECT    十字形：MORPH_CROSS      椭圆形：MORPH_ELLIPSE;
#参数ksize:是内核的尺寸(n,n)
#参数anchor:锚点的位置

   
