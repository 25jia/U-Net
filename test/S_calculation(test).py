from PIL import Image
import numpy as np
#from train import mask
import glob
import os
import cv2

path1 = 'mask/*.png'  #测试

path2 = 'true_label/*.png'   #手绘标签

print('求耕地面积占比,测试分析模型特点\n')

def whitepercent(a):
    c=2*np.abs(a-0.5)
    s=c.sum()  #s=sum(sum(c))
    n=a.sum()   #n=sum(sum(a))
    p=n/s
    return p

for files in glob.glob(path1):    #原始耕地面积生成
    filepath,filename = os.path.split(files)
    a = cv2.imread(files, -1)
    m  = a.max()
    a = a/m   #a = a/255
    a[a>=0.5] = 1
    a[a<0.5] = 0
    cv2.imwrite('bi/'+filename,a*255)
    print(filename+'      ' +' %.8f \n' %whitepercent(a))

for files in glob.glob(path2):   #tif耕地面积计算
    filepath,filename = os.path.split(files)
    a = cv2.imread(files, -1) /255#np.asarray(Image.open(files))
    print(filename+'      ' +' %.8f \n' %whitepercent(a))
