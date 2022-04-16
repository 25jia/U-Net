import os
import glob
import cv2
import numpy as np


def mask_together(maskpath, savepath):  # (6*6)---->6
   k = 0
   a = [ ]
   for k in range (0,36):
        file = maskpath+"/%d"%k+".png"
        i = k%6
        filepath,filename = os.path.split(file)
        print(filename)
        x = cv2.imread(file, -1)
        #print(x)
        a.append(x)   #图片进表
        if (i==5) : #将图片回归原方向
           for o in range(0,6):
               u = a[o]
               #cv2.imwrite("predict/out12change"+"/"+ "%d"%(o+k-5)+".png", a[o])  #中间过程 
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


   y = x[0]
   x[0] =y[:,  :214]    #这里修改
   y = x[1]
   x[1] = y[: ,  : 130]
    #左右合并
   line1 = np.concatenate([x[0], x[1]], axis = 1)
   line1 = np.concatenate([line1, x[2]], axis = 1)
   line1 = line1[0 : 244, :]

   y = x[3]
   x[3] =y[:,  :214]
   y = x[4]
   x[4] = y[: ,  : 130]#这里修改
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
    for files in glob.glob(file_path+"/*_bi.png"): #遍历image每个文件,后缀随前程序为_ori
        i+=1
        filepath,filename = os.path.split(files)
        x = cv2.imread(files, -1)/255
        x[x>=0.5] = 1
        x[x<0.5] = 0
        cv2.imwrite(file_savepath+'/'+filename+".tif", x) #转png
    print('   new_tif_number = '+'%d'%i)

   
#路径是数据文件夹相对路径

#tifpath1 =  r"predict/tif_question"  #存放原始卫星tif文件夹
#in1 = r"predict/in1" 
#in6 =  r"predict/in6"
#in12 = r"predict/in12"

out12 = r"hand_out12"
out6 =  r"hand_out6"
out1 = r"hand_out1"

tifpath2 = r"hand_tif_result"  #存放标注结果tif文件夹

#n = png_generator(tifpath1, in1 )#tif图片数目

#for number in range(0,n):
    #test_cut(in1+'/%d.png'%number, in6)
    #test_generator(in6,  in12)
    #test_result(in12, out12, 72)
mask_together(out12, out6)
result(out6, out1, '_ori', 0) #灰度图结果
result(out6, out1, '_bi', 0) #二值图预测结果
print('finish_predict %d / %d'%(1, 1))

tif_generator(out1, tifpath2)
