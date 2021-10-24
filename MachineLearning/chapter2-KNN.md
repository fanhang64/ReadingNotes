### K - 近邻算法

#### 2.1 k-近邻算法概述

##### 2.1.1 kNN概述

K-近邻算法采用**测量不同特征值之间的距离**进行分类。

**优点：** 精度高，对异常数据不敏感，无数据输入假定。

**缺点：** 计算复杂度高，空间复杂度高。

**使用数据范围：** 数值型(回归)和标称型。

**工作原理：** 存在一个样本数据集合，也称作训练样本集，并且样本集合每个数据都存在标签，即我们知道样本集中每一数据与所属分类的对应关系。输入没有标签的新数据，将新数据的每个特征和样本集中数据对应的特征进行比较，然后算法提取样本集中特征最相似数据（最近邻）的分类标签。一般来说只选择样本数据集中前K个最相似的数据。

**k-近邻算法的一般流程**

1. 收集数据
2. 准备数据
3. 分析数据：可以使用多重方式对数据进行分析，例如使用matplotlib进行数据可视化
4. 训练数据：不适用KNN算法
5. 测试算法：计算错误率
6. 使用算法：错误率在可以接受的范围内, 就可以运行k近邻算法进行分类。



举个简单的例子，我们可以使用k-近邻算法分类一个电影是爱情片还是动作片。

![KNN-1](https://gitee.com/fanhang64/my_images/raw/master/2021/KNN-1.png)

如图是我们已有的数据集合，也就是训练样本集，这个数据集有两个特征，打斗镜头和接吻镜头，我们也知道每个电影的所属类型，即分类标签。以我们的经验，接吻镜头多，是爱情片，打斗镜头多是动作片。现在我给你一部电影的打斗镜头和接吻镜头，不告诉你电影类型，你能够根据我给的信息来判断是什么类型的电影。KNN算法也可以像我们人一样做到这点。

##### 2.1.2 距离测量

KNN算法根据特征比较，然后提取样本集中特征最相似数据的分类标签。如下如图

![KNN-2](https://gitee.com/fanhang64/my_images/raw/master/2021/KNN-2.jpg)

我们可以从散点图大致推断，这个红点标记可能属于动作片，因为距离那个动作片更近，KNN算法使用距离度量来进行判断。这个电影分类有两个特征值，也就是2维实数的向量空间，可以使用中学学到的两点间距离公式。

![KNN-3](https://gitee.com/fanhang64/my_images/raw/master/2021/KNN-3.jpg)

通过计算，我们可以得到如下的结果。

- (101, 20) 到 动作片(108,5)的距离约为16.55
- (101, 20) 到 动作片(115,8)的距离约为18.44
- (101, 20) 到 爱情片(5,89)的距离为118.22
- (101, 20) 到 爱情片(1, 101)的距离为128.69

通过计算可知电影到动作片(108, 5)的距离最近，为16.55 。如果根据这个结果判断是动作片，则这个算法是最近邻算法，而非K-近邻算法。

**K近邻算法步骤如下：**

1. 计算已知类别数据集中的点与当前点的距离；
2. 按照距离递增次序排序；
3. 选取与当前点距离最小的k个点；
4. 确定前K个点所在类别的出现频率；
5.  返回前k个点所出现频率最高的类别作为当前点的预测分类。



##### 2.1.3 示例：电影分类

**准备: 使用Py导入数据**

```python
# KNN.py

import numpy as np 
import operator

def createDataSet():
    group = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])
    labels = ['爱情片', '爱情片', '动作片', '动作片']
    return group, labels

```

**实施KNN算法**

根据两点距离公式，计算距离，选择距离最小的前k个点，并返回分类结果。

```python
# KNN.py
# KNN算法，分类器
"""
params:
	inX: 用于分类的数据（测试集）
	dataSet: 用于训练的数据(训练集)
	labels: 分类标签
	k: knn算法参数，选择距离最近的k个点
returns:
	sortedClassCount: 分类结果
"""
def classify0(intX, dataSets, labels, k=3):
    dataSetSize = dataSets.shape[0]  # dateset的shape(4, 2)
    diffMat = np.tile(intX, (dataSetSize, 1)) - dataSets # 数组沿各个方向复制, 在列方向重复intX1次，在行向量重复intX共dataSetSize次, 并计算差值
    print(diffMat)
    sqDiffMat = diffMat **2
    print(sqDiffMat)
    # sum()所有元素相加，sum(0)行上每个元素相加，例如：第一行第一个元素和第二行第一个元素+第三行第一列元素为第一个结果，sum(1)列上每个元素相加，第一行所有列相加为第一个结果
    # np.sum  默认所有元素相加， axis=0 
    distances = sqDiffMat.sum(axis=1)
    print(distances)
    # 开平方求出距离
    distances = distances**0.5
    sordedDistINdices = distances.argsort()  # 返回排序后的下标
    classCount = {}
    for x in range(k):
        # 取出前k个元素的类别
        voteIlabel = labels[sordedDistINdices[x]]
        print(voteIlabel, classCount)
        # 计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # key=operator.itemgetter(1)根据字典的值进行排序
    # key=operator.itemgetter(0)根据字典的键进行排序
    # f = operator.itemgetter(1)  => f([1, 2, 3]) => 2
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) 
    return sortedClassCount[0][0]  # 返回发生频率最高的标签

if __name__ == "__main__":
    #创建数据集
    group, labels = createDataSet()
    #测试集
    test = [101,20]
    #kNN分类
    test_class = classify0(test, group, labels, 3)
    #打印分类结果
    print(test_class)
```

**总结：**
KNN算法分类器并不会得到百分百正确的结果，我们可以使用多种方法检测分类器的正确率。此外分类器的性能也会受到多种因素的影响，如分类器设置和数据集等。不同的算法在不同数据集上的表现可能完全不同。为了测试分类器的效果，我们可以使用已知答案的数据，当然答案不能告诉分类器，检验分类器给出的结果是否符合预期结果。通过大量的测试数据，我们可以得到分类器的错误率-分类器给出错误结果的次数除以测试执行的总数。**错误率**是常用的评估方法，**主要用于评估分类器在某个数据集上的执行效果**。完美分类器的错误率为0，最差分类器的错误率是1.0。同时，我们也不难发现，**k-近邻算法没有进行数据的训练**，直接使用未知的数据与已知的数据进行比较，得到结果。因此，可以说k-近邻算法不具有显式的学习过程。

#### 2.2 示例：使用K-NN算法改进约会网站的配对效果

##### 2.2.1 准备数据

约会数据1000行，每行一个样本，每个样本3个特征值，包括：每年飞行里程数，玩视频游戏百分比，每周消费冰激凌公升数。

从`datingTestSets.txt`文件中读取样本，转为合适处理的格式。

```python
def file2matrix(filename):
    """
        params:
            filename: 文件名KNN-3
        retruns:
            returnMat: 特征矩阵
            classLabelVector: 分类label向量
    """
    fr = open(filename)
    # 读取文件的所有内容
    arrayLines = fr.readlines()
    # 获取文件的行数
    numberOfLines = len(arrayLines)
    # 返回的特征矩阵
    returnMat = np.zeros((numberOfLines, 3))
    # 返回的分类标签
    classLabelVector = []
    # 行索引
    index = 0
    for line in arrayLines:
        line = line.strip()
        listFromLine = line.split("\t")
        returnMat[index, :] = listFromLine[:3]
        #根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        else:
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector

if __name__ == "__main__":
    mat, classLabels = file2matrix("datingTestSet.txt")
    print(mat, classLabels)
```

##### 2.2.2 分析数据：使用matplotlib创建散点图

数据可视化，使用`matplotlib`模块，制作原始数据的散点图。

```python

def show_datas(datingDataMat, datingLabels):
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:,1], datingDataMat[:,2], s=15*np.array(datingLabels), c=15*np.array(datingLabels))  # s控制散点大小，c控制颜色
    # plt.show()
    font = FontProperties(fname="/usr/share/fonts/droid/DroidSansFallbackFull.ttf", size=14)

    # 当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区, 不共享x轴和y轴,fig画布的大小为(13,8)
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13, 8))
    number_of_labels = len(datingLabels)
    labels_colors = []
    for x in datingLabels:
        if x == 1:
            labels_colors.append('black')
        elif x == 2:
            labels_colors.append('orange')
        else:
            labels_colors.append('red')
    # 画出散点图，以飞行历程和玩游戏数据画散点图，散点大小15， 透明度0.5
    axs[0][0].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 1], s=15, c=labels_colors, alpha=.5)        
    axs1_title_text = axs[0][0].set_title('每年获得飞行常客历程数与玩游戏时间占比', FontProperties=font)
    axs1_xlabel_text = axs[0][0].set_xlabel('每年获得飞行常客历程数', FontProperties=font)
    axs1_ylabel_text = axs[0][0].set_ylabel('玩游戏时间', FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')
    
    # 画出散点图，以飞行历程和冰激凌数据画散点图，散点大小15， 透明度0.5
    axs[0][1].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 2], s=15, c=labels_colors, alpha=.5)        
    axs1_title_text = axs[0][1].set_title('每年获得飞行常客历程数与消费冰激凌公升数', FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel('每年获得飞行常客历程数', FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel('每周消费冰激凌公升数', FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')
    
    # 画出散点图，以玩游戏数据和冰激凌消费公升数画散点图，散点大小15， 透明度0.5
    axs[1][0].scatter(x=datingDataMat[:, 1], y=datingDataMat[:, 2], s=15, c=labels_colors, alpha=.5)        
    axs1_title_text = axs[1][0].set_title('以玩游戏数据和每周冰激凌消费公升数', FontProperties=font)
    axs1_xlabel_text = axs[1][0].set_xlabel('以玩游戏数据', FontProperties=font)
    axs1_ylabel_text = axs[1][0].set_ylabel('每周冰激凌消费公升数', FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    # 设置图例
    didntlike = mlines.Line2D([], [], color='black', marker='.', markersize=6, label='didntlike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.', markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.', markersize=6, label='largeDoses')

    axs[0][0].legend(handles=[didntlike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntlike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntlike, smallDoses, largeDoses])

    plt.show()
```

结论：通过数据可以很直观的发现数据的规律，比如以玩游戏所消耗时间占比与每年获得的飞行常客里程数，只考虑这二维的特征信息，给我的感觉就是海伦喜欢有生活质量的男人。为什么这么说呢？每年获得的飞行常客里程数表明，海伦喜欢能享受飞行常客奖励计划的男人，但是不能经常坐飞机，疲于奔波，满世界飞。同时，这个男人也要玩视频游戏，并且占一定时间比例。能到处飞，又能经常玩游戏的男人是什么样的男人？很显然，有生活质量，并且生活悠闲的人。

##### 2.2.3 准备数据： 归一化数值

下表给出了四组样本，如果想要计算样本3和样本4 的距离，通过使用欧式距离公式计算：

![KNN-5](https://gitee.com/fanhang64/my_images/raw/master/2021/KNN-5.jpg)

计算方法如下图：

![KNN-4](https://gitee.com/fanhang64/my_images/raw/master/2021/KNN-4.jpg)

我们可以发现数值相差较大的属性对计算结果影响较大，也就是说每年飞行里程数多结果影响远大于其他两个特征，而产生这种现象的唯一原因，仅仅是因为飞行常客里程数远大于其他特征值。但海伦认为这三种特征是同等重要的，因此作为三个等权重的特征之一，飞行常客里程数并不应该如此严重地影响到计算结果。

**处理方式：** 对于这类数据，我们通过采用数值归一法,如将数值取值范围处理到0到1，或-1到1之间，先公式可以将任意取值范围的特征值转为0到1区间的值：

```
new_value = (old_value - min) / (max - min)

# min 和 max 代表数据集中最小的特征值和最大的特征值
```

**归一化示例:**

```python
def autoNorm(data_set):
    """对数据归一化
        params:
            data_set: 特征矩阵
        returns:
            norm_data_set: 归一化后的矩阵
            ranges: 数据范围
            min_val: 数据最小值 
        公式：
            newValue = (oldValue - min) / (max - min)
    """    
    min_val = data_set.min(axis=0)  # 按行取最小值，例如比较第一列的所有行
    max_val = data_set.max(axis=0)
    # 最大值和最小值的差
    ranges = max_val - min_val
    norm_data_set = np.zeros(np.shape(data_set))
    # 返回data_set的行数
    m = data_set.shape[0]
    # oldvalue - min
    norm_data_set = data_set - np.tile(min_val, (m ,1))

    norm_data_set = norm_data_set / np.tile(ranges, (m, 1))
    return norm_data_set, ranges, min_val
```

##### 2.2.4 测试算法

机器学习算法一个重要的工作就是评估算法的正确性，通常我们提供已有数据的90%作为训练样本来训练分类器，而使用其余的10%数据去测试分类器，检测分类器的正确率。需要注意的是10%的测试数据应该是随机选择的。

这里我们使用错误率来检测分类器的性能，对于分类器来说，错误率就是分类器给出错误结果的次数除以测试数据的总数。

```python
def datingClassTest():
    """
        returns:
            normDataSet: 归一化后的特征矩阵
            ranges: 数据范围
            minVals: 最小值
    """
    filename = 'datingTestSet.txt'
    datingMat, datingLabels = file2matrix(filename)
    hoRatio = 0.1

    # 数据归一化
    normDataSet, ranges, minVals = autoNorm(datingMat)
    # 获取行数
    m = normDataSet.shape[0]  # 1000
    # 测试数据的个数
    numTestVecs = int(m * hoRatio)  # 1000 * 0，1 即10%测试数据 100 
    # 错误计数器
    errorCount = 0
    for x in range(numTestVecs):  # 0 - 99
        # normDataSet[x,:] 一行数据即一个样本,前10%数据为测试数据
        # normDataSet[numTestVecs:m,:] 其余90%为训练数据
        classfierResult =classify0(normDataSet[x,:], normDataSet[numTestVecs:m,:], datingLabels[numTestVecs:m], k=4)
        print("分类结果：%d, 真实结果: %d" %(classfierResult, datingLabels[x]))
        if classfierResult != datingLabels[x]:
            errorCount += 1
    print("错误率：%.2f" % ((errorCount / float(numTestVecs)) * 100))
```

##### 2.2.5 使用算法

通过该程序海伦会在约会网站上找到某个人并输入他的信息。程序会给出她对男方喜欢程度的预测值。

```python
def classifyPerson():
    # 输出结果
    result_list = ['讨厌', '有些喜欢', '非常喜欢']
    # 三维特征
    precentTats = float(input("玩视频游戏所耗时间百分比:"))
    ffMiles = float(input("每年获得的飞行常客里程数:"))
    iceCream = float(input("每周消费的冰激淋公升数:"))
    filename = 'datingTestSet.txt'
    datingDataMat, datingLabels = file2matrix(filename)
    # 数据归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)  # 所有列都进行归一化
    # 生成NumPy数组,测试集
    inArr = np.array([ffMiles, precentTats, iceCream])
    # 归一化
    norminArr = (inArr - minVals) / ranges
    # 返回分类结果
    classfierResult = classify0(norminArr, normDataSet, datingLabels, k=3)
    print("海伦可能%s这个人" % result_list[classfierResult-1])
```

#### 2.3 示例：手写识别系统

对于需要识别的数字已经使用图形处理软件，处理成具有相同的色彩和大小：宽高是32像素x32像素。尽管采用本文格式存储图像不能有效地利用内存空间，但是为了方便理解，我们将图片转换为文本格式，数字的文本格式如图所示。

![KNN-6](https://gitee.com/fanhang64/my_images/raw/master/2021/KNN-6.jpg)

##### 2.3.1 准备数据

这里使用sklearn模块提供的KNN算法

```PYTHON
def img2vector(filename):
    """将32x32的二进制的图像转为1x1024的向量
        
        params:
            filename: 文件名
        returns:
            returnVec: 返回的向量
    """

    returnVec = np.zeros((1,1024))
    fr = open(filename)    
    for x in range(32):  # 行
        lineStr = fr.readline()
        print(len(lineStr))
        for j in range(32):  # 列
            returnVec[0, 32*x+j] = int(lineStr[j])
    return returnVec

if __name__ == "__main__":
    r = img2vector('trainingDigits/0_0.txt')
    print(r)
    print(r.shape)
```



##### 2.3.2 测试算法

```python
import os
import numpy as np
from KNN import classify0

from sklearn.neighbors import KNeighborsClassifier as KNN

def img2vector(filename):
    """将32x32的二进制的图像转为1x1024的向量
        
        params:
            filename: 文件名
        returns:
            returnVec: 返回的向量
    """
    returnVec = np.zeros((1,1024))
    fr = open(filename)    
    for x in range(32):  # 行
        lineStr = fr.readline()
        for j in range(32):  # 列
            returnVec[0, 32*x+j] = int(lineStr[j])
    return returnVec

def handWritingClassTest():
    # 测试集labels
    hwLabels = []
    # 返回trainingDigits目录下的文件名
    trainingFileList = os.listdir('trainingDigits')
    # 文件个数
    m = len(trainingFileList)
    # 初始化训练集矩阵, 一个文件为一行数据，一个样本
    trainingMat = np.zeros((m, 1024))
    # 从文件中解析训练集类别
    for i in range(m):
        fileStr = trainingFileList[i]
        # 获得分类的数字
        classNumber = int(fileStr.split('_')[0])
        hwLabels.append(classNumber)  # labels
        # 将每个文件添加到trainingMat中
        trainingMat[i, :] = img2vector(f'trainingDigits/{fileStr}')
    print(trainingMat)
    neigh = KNN(n_neighbors=3, algorithm='auto')
    # 拟合模型
    neigh.fit(trainingMat, hwLabels)
    # 测试数据
    testFileList = os.listdir('testDigits')
    # 错误检测技术
    errorCount = 0
    # 测试数据数量
    mTest = len(testFileList)
    for x in range(mTest):
        fileStr = testFileList[x]
        # 获得分类的数字
        classNumber = int(fileStr.split('_')[0])
        # 获取测试即向量用于测试。
        vectorUnderTest = img2vector(f'testDigits/{fileStr}')
        # 获得测试结果
        # classfierResult = classify0(vectorUnderTest, trainingMat, hwLabels, k=3)
        classfierResult = neigh.predict(vectorUnderTest)  # 使用sklearn模块
        print("返回的结果为%d，真实的结果为%d" % (classfierResult, classNumber))
        if classfierResult != classNumber:
            errorCount += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount/mTest * 100))

if __name__ == "__main__":
    r = img2vector('trainingDigits/0_0.txt')
    print(r)
    print(r.shape)
    handWritingClassTest()
```

**总结：**

**优点**

- 简单好用，容易理解，精度高，理论成熟，既可以用来做分类也可以用来做回归；
- 可用于数值型数据和离散型数据；
- 训练时间复杂度为O(n)；无数据输入假定（？）；
- 对异常值不敏感（？）

**缺点**

- 计算复杂性高；空间复杂性高；
- 样本不平衡问题（即有些类别的样本数量很多，而其它样本的数量很少）；
- 一般数值很大的时候不用这个，计算量太大，可能很耗时。但是单个样本又不能太少，否则容易发生误分。
- 最大的缺点是无法给出数据的内在含义。

