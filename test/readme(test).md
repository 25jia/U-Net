<Br/><S_calculation(test).py>
<Br/>用以计算test结果不同个归一化情形面积占比，以确定归一化形式，生成二值化图片保存于bi文件夹中。
<Br/><result未优化.py>
<Br/>用于生成未经过“6图平均值”，也就是predict.py中的算法优化的预测结果，用于和优化后的结果对比。生成过程通过对应图片拼接生成
<Br/>************************************************************
<Br/>之前运行结果已存在于对应的文件夹中。
<Br/>可供参考。
<Br/>************************************************************
<Br/>image:测试集，不同于训练集裁剪方式,256*256
<Br/>mask：预测标签原图,256*256
<Br/>bi：二值化后预测标签图,256*256
<Br/>true_label:对应人工标记标签,256*256
<Br/>result未优化：未优化产生的8张测试结果图，500*600
