import os
import glob
import cv2
import numpy as np

#将二值png转tif
def tif_generator():
    print(' <.png to .tif >'+':')
    i=0
    for files in glob.glob("*.png"): #遍历image每个文件,后缀随前程序为_ori
        i+=1
        filepath,filename = os.path.split(files)
        x = cv2.imread(files, -1)/255
        x[x>=0.5] = 1
        x[x<0.5] = 0
        print(x.shape)
        cv2.imwrite(filename+".tif", x) #转png
    print('   new_tif_number = '+'%d'%i)

tif_generator()
