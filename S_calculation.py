from PIL import Image
import numpy as np
#from train import mask
import glob
import os
import cv2

path1 = 'data/tif/train_mask/*.tif'

path2 = 'predict/tif_result/*.tif'
path3 = 'predict/out1/*_bi.png'  #人工核实png成功生成，png格式要'/255'

print('求耕地面积占比\n')

def whitepercent(a):
    c=2*np.abs(a-0.5)
    s=c.sum()  #s=sum(sum(c))
    n=a.sum()   #n=sum(sum(a))
    p=n/s
    return p

for files in glob.glob(path1):    #原始耕地面积生成
    filepath,filename = os.path.split(files)
    a = cv2.imread(files, -1) #np.asarray(Image.open(files))
    print(filename+'      ' +' %.8f \n' %whitepercent(a))

for files in glob.glob(path2):   #tif耕地面积计算
    filepath,filename = os.path.split(files)
    a = cv2.imread(files, -1) #np.asarray(Image.open(files))
    print(filename+'      ' +' %.8f \n' %whitepercent(a))

for files in glob.glob(path3):  #png耕地面积计算
    filepath,filename = os.path.split(files)
    a = cv2.imread(files, -1)/255 #np.asarray(Image.open(files))
    print(filename+'      ' +' %.8f \n' %whitepercent(a))
