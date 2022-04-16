
import os
import glob
import cv2
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec

#目标：将test图片切分成不重合的小块，分别进行：
#   正、左、右、下和它们的镜像8种操作，
#   再进入模型预测，结果还原求最大概率:
#       结果求和，>=4置1，<4置0

def Mirror_x(img):                #关于x轴镜像
   new  = cv2.flip(img,  0)  
   return new

def Mirror_y(img):                #关于y轴镜像
    new  = cv2.flip(img,  1)
    return new

def  AntiClock90(img):               #逆时针旋转90度
    trans = cv2.transpose(img) #矩阵转置
    new  = cv2.flip(trans,  0)  #关于x轴镜像
    return new

def  Clock90(img):               #顺时针旋转90度
    trans = cv2.transpose(img) #矩阵转置
    new  = cv2.flip(trans,  1)  #关于y轴镜像
    return new

def png_generator(file_path, file_savepath):
    print(' <.tif to .png >'+file_path+'-->'+file_savepath+':')
    i=0
    for files in glob.glob(file_path+"/*.tif"): #遍历image每个文件
        i+=1
        filepath,filename = os.path.split(files)
        print(filename+'   '+'%d'%(i-1)+".png")
        x = cv2.imread(files, -1)
        m = x.max()
        q = (x*255)/m #标准化
        cv2.imwrite(file_savepath+'/'+'%d'%(i-1)+".png", q) #转png
    print('   new_png_number = '+'%d'%i)
    return i
#将图片(500,600)转为(256,256)2*3=6张
    
def test_cut(img, savepath):
    print('test_cut')
    print(img+'-->'+savepath+':')
    k=0
    a = [0, 244]
    b = [0, 214, 344]
    for p in range(0,2):  #竖着5张
        for q in range(0,3):  #横着9张   
                k+=1
                #filepath,filename = os.path.split(files)
                i = a[p] 
                j = b[q]
                x = cv2.imread(img, -1)
                m = x[i : (256+i), j : (256+j) ]
                cv2.imwrite(savepath+"/"+'%d'%(k-1)+".png", m)
    print('   number = '+'%d'%k) #2*3 = 6
    
def test_generator(imgpath, savepath):   #6---->(6*6)
   k=0
   for tt in range (0,6):
      file = imgpath+"/%d"%tt+".png"
      filepath,filename = os.path.split(file)
      print(filename)
      #filepath,filename = os.path.split(files)
      img = cv2.imread(file, -1)

      cv2.imwrite(savepath+"/"+"%d"%(0 + 6*k)+".png", img)
      cv2.imwrite(savepath+"/"+"%d"%(1 + 6*k)+".png", Mirror_x(img))
      cv2.imwrite(savepath+"/"+"%d"%(2 + 6*k)+".png", Mirror_y(img))
      
      img = AntiClock90(img)
      img = AntiClock90(img)
      cv2.imwrite(savepath+"/"+"%d"%(3 + 6*k)+".png", img)
      cv2.imwrite(savepath+"/"+"%d"%(4 + 6*k)+".png", Mirror_x(img))
      cv2.imwrite(savepath+"/"+"%d"%(5 + 6*k)+".png", Mirror_y(img))
#优化后保留0和180旋转以及其x,y镜像
      #img = AntiClock90(img)
      #cv2.imwrite(savepath+"/"+ "%d"%(6 + 12*k)+".png", img)
      #cv2.imwrite(savepath+"/"+"%d"%(7 + 12*k)+".png", Mirror_x(img))
      #cv2.imwrite(savepath+"/"+"%d"%(8 + 12*k)+".png", Mirror_y(img))

      #img = AntiClock90(img)
      #cv2.imwrite(savepath+"/"+"%d"%(9 + 12*k)+".png", img)
      #cv2.imwrite(savepath+"/"+"%d"%(10 + 12*k)+".png", Mirror_x(img))
      #cv2.imwrite(savepath+"/"+"%d"%(11 + 12*k)+".png", Mirror_y(img))
      
      k+=  1
   print('   number = '+'%d'%(k*6)) #12*6 = 72

from model import *
from data import *
def test_result(path, savepath, n = 36):
   # python test.py
   """
   注：
           A: target_size()为图片尺寸，要求测试集图像尺寸设置和model输入图像尺寸保持一致，
                   如果不设置图片尺寸，会对输入图片做resize为处理，输入网络和输出图像尺寸默认均为（256,256），
           B: 且要求图片位深为8位，24/32的会报错！！
           C: 测试集数据名称需要设置为：0.png……
           D：model.predict_generator( ,n, ):n为测试集中样本数量，需要手动设置，不然会报错！！
   """
   # 输入测试数据集，
   testGene = testGenerator(path , n,target_size = (256,256)) # data
   # 导入模型
   model = unet(input_size = (256,256,1)) # model 
   # 导入训练好的模型
   model.load_weights("unet_model_300.hdf5")
   # 预测数据
   results = model.predict_generator(testGene, n, verbose=1) # keras
   #print(results)
   saveResult(savepath, results) # data
   print("over")

def mask_together(maskpath, savepath):  # (6*12)---->6
   k = 0
   a = [ ]
   for k in range (0,36):
        file = maskpath+"/%d"%k+".png"
        i = k%6
        filepath,filename = os.path.split(file)
        print(filename)
        x = cv2.imread(file, -1)
        a.append(x)   #图片进表
        if (i==5) : #将图片回归原方向
           a[1] = Mirror_x(a[1])
           a[2] = Mirror_y(a[2])
        
           a[3] = Clock90(a[3])
           a[3] = Clock90(a[3])
           
           a[4] = Mirror_x(a[4])
           a[4] = Clock90(a[4])
           a[4] = Clock90(a[4])
           
           a[5] = Mirror_y(a[5])
           a[5] = Clock90(a[5])
           a[5] = Clock90(a[5])
        
          
           for o in range(0,6):
               u = a[o]
               cv2.imwrite("predict/out12change"+"/"+ "%d"%(o+k-5)+".png", a[o])  #中间过程
               
               u =a[o]
               u = u/(u.max())   #每张图最大值归一化
               #u[u<=0.5] *= 0.5
               #u[u>0.5] *= 1.5
               u = u-0.5     #为了求和黑的更黑白的更白
               a[o] = u                   
               #u[u>=0] = 1   #此除优先二值化，可以保留细节特征，但是整体图像效果下降
               #u[u<0] = -1
           
           mask = a[0]+a[1]+a[2]+a[3]+a[4]+a[5]
           mask = mask/6 + 0.5    #取平均，恢复（0,1）
           
           maskmax = mask.max()   #最大值归一化
           mask = mask/maskmax
           
           write1 = mask*255
           name = (k-5)/6 #文件名
           
           cv2.imwrite(savepath+"/"+ "%d"%name+"_ori.png", write1)
           
           mask[mask>0.5]=255 
           mask[mask<=0.5]=0
           cv2.imwrite(savepath+"/"+ "%d"%name+"_bi.png", mask)
           a = [ ] #列表更新
           print('\n')
           print('   finish  '+'%d'%(name)+'/6')

def result(maskpath, savepath, typename, namenumber):  #6---->1
   i = 0
   x = [ ]
   for k in range(0,6):
      file = maskpath+"/%d"%k+typename+".png"
      filepath,filename = os.path.split(file)
      print(filename)
      x.append(cv2.imread(file, -1))
      i+=1

   y = x[1]   
   x[1] = y[: , 42 : 130]
    #左右合并
   line1 = np.concatenate([x[0], x[1]], axis = 1)
   line1 = np.concatenate([line1, x[2]], axis = 1)
   line1 = line1[0 : 244, :]

   y = x[4]
   x[4] = y[: , 42 : 130]
   #左右合并
   line2 = np.concatenate([x[3], x[4]], axis = 1)
   line2 = np.concatenate([line2, x[5]], axis = 1)
   mask = np.vstack((line1, line2))   #上下合并

   print(mask.shape)
   cv2.imwrite(savepath+"/%d"%namenumber +typename+ ".png", mask)
   print('\n')

#将二值png转tif
def tif_generator(file_path, file_savepath):
    print(' <.png to .tif >'+file_path+'-->'+file_savepath+':')
    i=0
    for files in glob.glob(file_path+"/*_bi.png"): #遍历image每个文件,后缀随前程序为_bi
        i+=1
        filepath,filename = os.path.split(files)
        x = cv2.imread(files, -1)/255
        x[x>=0.5] = 1
        x[x<0.5] = 0
        cv2.imwrite(file_savepath+'/'+filename+".tif", x) #转png
    print('   new_tif_number = '+'%d'%i)

   
#路径是数据文件夹相对路径

tifpath1 =  r"predict/tif_question"  #存放原始卫星tif文件夹
in1 = r"predict/in1" #
in6 =  r"predict/in6"
in12 = r"predict/in12"

out12 = r"predict/out12"
out6 =  r"predict/out6"
out1 = r"predict/out1"

tifpath2 = r"predict/tif_result"  #存放标注结果tif文件夹

n = png_generator(tifpath1, in1 )#tif图片数目

for number in range(0,n):
    test_cut(in1+'/%d.png'%number, in6)
    test_generator(in6,  in12)
    test_result(in12, out12, 36)
    mask_together(out12, out6)
    result(out6, out1, '_ori', number) #灰度图结果
    result(out6, out1, '_bi', number) #二值图预测结果
    print('finish_predict %d / %d'%(number+1, n))

tif_generator(out1, tifpath2)
