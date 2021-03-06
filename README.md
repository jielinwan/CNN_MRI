# CNN_MRI
我们使用了3dcnn对脑部mri图像进行分类，分类标准为是否酗酒。
## 数据描述
我们使用的数据集包括505位病人在4年间的脑部MRI图像，分为有5个label：label 0，label 1，label 2，label 3，label 4。label n表示从倒数第n年开始酗酒，label 0表示不酗酒。
## 数据预处理
由于在酗酒前的脑图是相同的，酗酒后的脑图也是相同的。例如，对于label 2的人来说，前面2年不酗酒的MRI图像是相同的，后面两年酗酒的MRI图像也是相同的。因为我们需要对重复的MRI图像进行去重筛选。

为了使我们的模型具有更强的泛化能力，以及扩充我们的数据集，我们对MRI图像中添加了均值为0，方差为6和12的两种噪声。

在磁共振成像(MRI)的纹理分析和定量分析中出现的一个问题是，提取的结果在连续扫描或重复扫描之间，或者在同一扫描中，在不同的解剖区域之间是不可比较的。原因在于由于MRI仪器的使用，会导致了扫描内和扫描间图像强度的变化。因此，在进行进一步的图像分析之前，应该对MRI图像使用图像强度归一化方法。在这里，我们首先对数据进行了一个归一化处理，使得MRI图像中数值处于标准正态分布。

## 训练集的划分
按照分层抽样的思想，我们按照各个label来对训练集、测试集进行划分。在每一个label中首先对label中的病人划分成五等分，抽取了一份作为测试集，其余4份作为训练集。然后，对于训练集的病人会把没有加入噪声的MRI图和加了噪声的MRI图都加入训练集，对于训练集的病人只会加入没有添加噪声的MRI图。也就是说我们在训练时会使用添加了噪声的数据，在测试的时候不会使用添加了噪声的数据。

## main_32.py 模型描述
我使用的模型包括以下几个部分：

卷积层 conv
池化层 pool
全连接层 dense

具体结构如下：

    conv-32
    max_pool
    
    conv-64    
    max_pool 
    
    conv-128    
    max_pool  
    
    (dropout-0.85)
    dense-128 
    (dropout 0.85)   
    dense-32   
    (dropout 0.85) 
    dense-2         
    
    模型的学习率为0.0003
## FileAddNoise.py
此文件对源数据进行添加噪声和归一化。生成的文件为添加均值为0，方差为0 3 6 9 12的数据，并且已经进行了归一化。

## group1.mat
此文件为源数据中的label集

## Result_32
此文件夹中包含了main_32.py的运行结果，包括了在每一个epoch上的训练集和测试集上的精度变化图(Accuracy_VS_Epoch.png),每一个epoch上的训练集和测试集的精度(Run_Report.txt),每一个epoch上的loss变化图(TrainLoss_VS_Epoch.png),以及具体数据的.npy文件。

在模型的结果上，最终训练集的精度趋向于1，测试集的精度在0.75左右。


