# Mathorcup2020-2021

<Br/>点击下方可查看该项目对应的论文
<Br/>[初赛论文](https://25jia.github.io/B1248-initial-xjl.pdf)
<Br/>[复赛论文](https://25jia.github.io/B1248-final-xjl.pdf)

<Br/>下面介绍程序思想以及程序运行方法
<Br/>我们基于U-Net卷积神经网络建立模型对题目数据进行处理，实现了精确度较高的遥感图像耕地地块分割与提取，成功计算出耕地面积比例并制作出较为清晰的耕地标签图。参考“集成学习”的思想，我们提出“多图平均二值法”用以辅助U-Net模型预测。该方法成功降低了预测结果中的噪声干扰，显著提升了预测的准确度。
我们进一步探究“多图平均二值法”的合理性与使用条件，并依据其原理改进了数据增广方法和模型训练参数，对数据增广的随机旋转范围和训练迭代次数进行优化，使得训练出的U-Net模型预测精度显著提升。优化后的模型准确率达99.02%，损失率0.041，在遥感图像空间分辨率为2m的条件下，测试中可作出误差在2%范围内的对耕地的精确分割。同时，为了在保证预测精度的同时提升耕地标签图的视觉效果，此外，我们采用腐蚀膨胀操作对预测结果做进一步处理。结果显示，腐蚀与膨胀操作使得模型预测效果得到进一步提升。相比初赛模型，本文模型具有更为优秀的分割性能和更为良好的视觉效果。本文模型的预测图像耕地边缘更为清晰，噪声基本消失，道路连通性强。利用上述优化模型，本文预测Test3，Test4图像的耕地地块分布并制作标签图，计算耕地面积占比。
<Br/>***************************************************************************
<Br/>我们的程序测试主要通过一下步骤完成：
<Br/>a)将tif卫星图放至‘pridict/tif_question’文件夹中
<Br/>b)运行predict.py程序，可以在out1里看到标签(二值化前ori后bi)，tif_result里面是相应二值化图片的tif
<Br/>c)运行S_calculation.程序计算面积占比，适当修改地址路径
<Br/>d）运行predict前请检查文件夹in1、out1、tif_result为空，以防影响结果查看。
<Br/>e）按照predict文件夹下readme以及其中erode&dilate_Enhance_view文件夹的read指示操作，获得进一步腐蚀膨胀优化的结果
<Br/>腐蚀膨胀程序使用方法简述：
<Br/>1将待操作的500x600 png格式图放在out1文件夹中；
<Br/>2修改<erode&dilate.py>中路径文件名及相关参数，运行，运行结果会以png格式保存在本文件夹中；
<Br/>3将需要转tif的png图片移动到answer_tif_generater文件夹中；
<Br/>4依次运行answer_tif_generater文件夹中的<answer_tif.py>,<answer_S.py>获得tif标签图及耕地面积；
<Br/>【注意事项】
<Br/>1算法输出图片名称标注'数字_bi.png.tif'，其中‘数字_bi.png’为其对应png文件名称；
<Br/>2mask将与in1中数字顺序对应，
<Br/>
<Br/>可能与tif_question顺序不一致，由于glob抓取；
<Br/>可以采用前位补零或按文件名修改算法等方式应对2中问题，程序将会输出tif与png文件名称对应关系，以及相应抓取顺序，希望您可以抽出时间核对;
<Br/>//补零//图片数为n位数，则补零对齐，如共100张则001，002,,,099,100编号;1000张则0001，0002,,,0011,0012,,,0111,0112,,,0999,1000编号

<Br/>3程序我们再本地Spider和PyCharm上成功运行，但由于设备不同，可能需在测试前对程序进行些许相应修改(地址，图片数目等参数)，希望您能谅解；
<Br/>4算法细节问题参见程序中的备注。
<Br/>5运行时，按照一下逻辑进行，其中中介文件夹反复使用，数据更新，不做保留。
<Br/>tif_question--->in1--->in6--->in12--->out12--->out12change--->out6--->out1--->tif_result
<Br/>问题及结果文件夹：tif_question、in1、out1、tif_result
<Br/>中介文件夹：in6--->in12--->out12--->out12change--->out6(最终记录为最后一张图片)
<Br/>6希望您能认真阅读本总说明以及每个文件夹中的readme文件(细节说明)，感谢！
<Br/>
<Br/>***************************************************************************
<Br/>【程序文件总体概况】
<Br/><data.py>:模型训练(train)和简单测试(test)时用的基本函数库//部分引自[1]
<Br/><model.py>:unet基本模型//部分引自[1]
<Br/><pre-data.py>:准备数据，用于将不可视的tif转化成可视的png处理，将image和mask剪切生成原始训练数据集
	       转化理由：过程可视化，方便寻找模型缺陷，设计合理应用方法，提高最终结果准确性
	       操作文件夹：data、train、test

<Br/><train.py>:训练网络，调用data.py和model.py //部分引自[1]
	操作文件夹：train
<Br/><test.py>:模型效果检测，测试集image和truelabel不同于原始训练数据集剪切，提供未经过优化的预测方法//部分引自[1]
	操作文件夹：test

<Br/><predict.py>:（详细见predict文件夹内readme文件）
<Br/>---1利用训练模型结果，完整进行由卫星tif至标签tif的转化，中间过程png格式可视化显示。
<Br/>---2提供优化后的预测方法：
<Br/>      a)将500*600切为256*256；
<Br/>      b)每256*256图通过180度旋转以及相应x，y镜像生成6张图输入模型;
<Br/>      c)将6图输出结果对比，利用适当算法求取平均，整合为1张256*256图;
<Br/>      d)将6张256*256图拼接成500*600的结果图;
<Br/>      操作文件夹：predict
<Br/>tif_question--->in1--->in6--->in12--->out12--->out12change--->out6--->out1--->tif_result
<Br/>【注】上述优化方法是通过观察下述方法生成的12图选定的6种合适图片处理方式完成。
<Br/><predict(12).py>:
<Br/>      a)将500*600切为256*256；
<Br/>      b)每256*256图通过90，180，270旋转以及相应x，y镜像生成12张图输入模型;
<Br/>      c)将12图输出结果对比，利用适当算法求取平均，整合为1张256*256图;
<Br/>      d)将6张256*256图拼接成500*600的结果图;
<Br/>      操作文件夹：predict
<Br/>tif_question--->in1--->in6--->in12--->out12--->out12change--->out6--->out1--->tif_result

<Br/><S_calculation>：利用tif或png标签图计算每张图耕地占比。

<Br/><erode&dilate.py>:腐蚀膨胀操作程序。
<Br/>***************************************************************************
<Br/>【问题解决方法简述】
<Br/>//1
<Br/>运行<S_calculation>，选择适当数据
或者运行‘问题1求标签耕地面积占比’文件夹下文件直接得到结果（初赛面积问题）
<Br/>//2
<Br/>按本说明，运行程序，形成结果
<Br/><pre-data.py>，<train.py>，<test.py>，<predict.py>
<Br/>//3
<Br/>根据预先test优化模型产生predict，寻找合适思路，获得合适标签图。
<Br/>//4
<Br/>设计合适腐蚀膨胀程序，对结果进一步优化
<Br/>//5
<Br/>计算耕地比例，适当修改计算面积程序的地址
<Br/>【注】您可以通过适当修改<S_calculation.py>的地址(原来的地址是没有经过腐蚀膨胀的预测图文件夹)或者使用predict文件夹下erode&dilate_Enhance_view文件夹下的<answer_S.py>计算耕地面积，我队实验使用<answer_S.py>计算的面积。
<Br/>
<Br/>***************************************************************************
<Br/>【创新点简述】
<Br/>1流程可视化
<Br/>2通过旋转、镜像输入方法，测试模型预测效果，选择模型优势预测角度，制定优化方法
<Br/>3多图平均二值法，在路径中断、点状误差、图像不清问题，提升优势明显；
<Br/>
<Br/>[注]上述平均法可通过二值化达到‘1+1>2’的效果，原因如下：
<Br/>平均的是非二值图，黑白部分在不同图片求和时相互抵消，强度随深浅而定。
<Br/>因此可做到类似“集成学习”模型取比例最高分类的效果。
<Br/>同时，该方法减少了训练模型所消耗的时间，明显减少了错误点数目。
<Br/>降低了损失，提高了准确率。
<Br/>
<Br/>4选择合适的batchsize和epoch，防止过拟合和较早进入饱和，降低了损失，提升了准确率。
<Br/>
<Br/>5将训练输入规范化256*256问题，适合多种图片，裁剪符合，稍改拼接程序即可进行预测。
<Br/>
<Br/>6根据“多图平均二值法”原理改进了数据增广方法和训练方法，对训练数据增广的随机旋转范围和训练迭代次数进行优化，使得训练出的U-Net模型预测精度显著提升。
<Br/>
<Br/>7为了在保证预测精度的同时提升耕地标签图的视觉效果，本文提出采用腐蚀膨胀操作对预测结果做进一步处理，详细分析了腐蚀膨胀操作的使用对面积精度的影响，选取合适操作处理图片，使得预测结果的视觉效果等得到显著提升。
<Br/>
<Br/>***************************************************************************
<Br/>【赛题运行案例】
<Br/>在该文件夹下，我们展示了程序运行案例和部分运行结果，供评委老师参考。
同时，也为我组论文提供论据支持。
<Br/>案例地址：
<Br/>链接：https://pan.baidu.com/s/15Z7avhCVuve5U8lt9Gsbkw 
<Br/>提取码：1248 
<Br/>
<Br/>***************************************************************************
<Br/>【参考文章来源简述】
<Br/>[1]https://blog.csdn.net/Xnion/article/details/105797671
