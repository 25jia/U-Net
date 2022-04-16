# data.py 训练测试时图片处理
from __future__ import print_function
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans

#adjustData()函数主要是对训练集的数据和标签的像素值进行归一化
def adjustData(img,mask):
    if(np.max(img) > 1):
        img = img / 255.0
        mask = mask /255.0
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)

    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    ''' 
# trainGenerator()函数主要是产生一个数据增强的图片生成器，不断生成图片
def trainGenerator(batch_size , train_path, image_folder, mask_folder, aug_dict, image_color_mode = "grayscale",
                    mask_color_mode = "grayscale", image_save_prefix  = "image", mask_save_prefix  = "mask",
                    flag_multi_class = False, num_class = 2, save_to_dir = None, target_size = (256,256),seed = 1):

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)#aug_dict控制处理范围和方法
    image_generator = image_datagen.flow_from_directory(
        train_path,#训练数据文件夹路径
        classes = [image_folder],#类别文件夹,对哪一个类进行增强
        class_mode = None,#不返回标签
        color_mode = image_color_mode,#灰度,单通道模式
        target_size = target_size,#转换后的目标图片大小
        batch_size = batch_size,#每次产生的进行转换后的图片张数
        save_to_dir = save_to_dir,#图片保存路径
        save_prefix  = image_save_prefix,#生成图片的前缀，提供save_to_dir时有效
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)#组合成一个生成器
    for (img,mask) in train_generator:#由于batch是4，所以一次返回两张，即img是一个4张灰度图片的数组，[4,256,256]
        img,mask = adjustData(img,mask)#数据和标签的像素值进行归一化，返回的img依旧是[4,256,256]
        yield (img,mask)#每次分别产出两张图片和标签
 
 
# testGenerator()函数主要是对测试图片进行规范，使其尺寸和维度上和训练图片保持一致
def testGenerator(test_path, num_image = 30, target_size = (256,256), flag_multi_class = False, as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img = img / 255.0
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape) #(1,)+(2,3) = (1,2,3)
		#将测试图片扩展一个维度，与训练时的输入[4,256,256]保持一致
        yield img
 
 
# geneTrainNpy()函数主要是分别在训练集文件夹下和标签文件夹下搜索图片，
# 然后扩展一个维度后以array的形式返回，是为了在没用数据增强时的读取文件夹内自带的数据
def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
	#相当于文件搜索，搜索某路径下与字符匹配的文件
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):#enumerate是枚举，输出[(0,item0),(1,item1),(2,item2)]
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
		#重新在mask_path文件夹下搜索带有mask字符的图片（标签图片）
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)#转换成array
    return image_arr,mask_arr
 
# 生成二值图片/灰度图： 1/255		
def saveResult(save_path,npyfile):
    for i, item in enumerate(npyfile):
        img = item[:,:,0]
        print(np.max(img), np.min(img))

        #img = img/255
        #img[img>=0.5]=1#此时1是浮点数，下面的0也是，灰度改成255
        #img[img<0.5]=0
        #print(np.max(img), np.min(img))
        io.imsave(os.path.join (save_path, "%d.png"%i), img)
