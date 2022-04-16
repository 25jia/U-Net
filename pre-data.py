import os
import glob
import cv2
import numpy as np

#将原始图片if(500,600)转为可视图片png(500,600)
def png_generator(file_path, file_savepath):
    print(' <.tif to .png >'+file_path+'-->'+file_savepath+':')
    i=0
    for files in glob.glob(file_path+"/*.tif"): #遍历image每个文件
        i+=1
        #filepath,filename = os.path.split(files)
        x = cv2.imread(files, -1)
        m = x.max() 
        q = (x*255)/m #标准化
        cv2.imwrite(file_savepath+'/'+'%d'%(i-1)+".png", q) #转png
    print('   new_png_number = '+'%d'%i)
 
def data_cut(file_path, file_savepath):
    print(' pre_train_cut:')
    print(file_path+'-->'+file_savepath+':')
    k=0
    for i in range(0, 245, 61):  #竖着5张
        for j in range(0, 345, 43):  #横着9张  
            for files in glob.glob(file_path+"/*.png"): #遍历image每个文件
                k+=1
                #filepath,filename = os.path.split(files)
                x = cv2.imread(files, -1)
                q = x[i : (256+i), j : (256+j) ]
                cv2.imwrite(file_savepath+"/"+'%d'%(k-1)+".png", q)
    print('   generater_number = '+'%d'%k) #5*9*8=360;5*9*2=90

def pre_test_cut(file_path, savepath):
    print('pre_test_cut :')
    print(file_path+'-->'+savepath+':')
    k=0  #顺序是依次右上每张图，再左切每张图。。。
    a = [0, 244]
    b = [0, 214, 344]
    for p in range(0,2):  #竖着2张
        for q in range(0,3):  #横着3张
            for files in glob.glob(file_path+"/*.png"): #遍历image每个文件
                k+=1
                filepath,filename = os.path.split(files)
                i = a[p] 
                j = b[q]
                x = cv2.imread(files, -1)
                m = x[i : (256+i), j : (256+j) ]
                cv2.imwrite(savepath+"/"+'%d'%(k-1)+".png", m)
    print('   generater_number = '+'%d'%(k)) #2*3 = 6, 6*8=48

#路径是数据文件夹相对路径

#tif path
image1 = r"data/tif/train_image"
mask1 =  r"data/tif/train_mask"
test1 = r"data/tif/test_image"

#tif-->png
#png save path
image2 =r"data/png/train_image"
mask2 = r"data/png/train_mask"
test2 = r"data/png/test_image"

#cut save path
image3 = r"train/image"
mask3 = r"train/mask"
test3 = r"test/image"
test4 =  r"test/true_label"

png_generator(image1, image2)
png_generator(mask1,  mask2)
png_generator(test1, test2)

data_cut(image2, image3)
data_cut(mask2,  mask3)

pre_test_cut(image2, test3)
pre_test_cut(mask2, test4)


