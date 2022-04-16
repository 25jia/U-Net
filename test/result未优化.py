import numpy as np
import cv2
import os

def result(maskpath, savepath,  namenumber):  #6---->1
   i = 0
   x = [ ]
   for k in range(0,6):
      file = maskpath+"/%d"%(k*8+namenumber)+".png"
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
   cv2.imwrite(savepath+"/%d"%namenumber+ ".png", mask)
   print('\n')

for i in range(0, 8):
    result('bi', 'result_ini', i)
