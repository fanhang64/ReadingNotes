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
3. 分析数据
4. 训练数据：不适用KNN算法
5. 测试算法：计算错误率
6. 使用算法



举个简单的例子，我们可以使用k-近邻算法分类一个电影是爱情片还是动作片。

![KNN-1](/home/fanzone/Documents/ReadingNotes/MachineLearning/KNN-1.png)

如图是我们已有的数据集合，也就是训练样本集，这个数据集有两个特征，打斗镜头和接吻镜头，我们也知道每个电影的所属类型，即分类标签。以我们的经验，接吻镜头多，是爱情片，打斗镜头多是动作片。现在我给你一部电影的打斗镜头和接吻镜头，不告诉你电影类型，你能够根据我给的信息来判断是什么类型的电影。KNN算法也可以像我们人一样做到这点。

##### 2.1.2 距离测量

KNN算法根据特征比较，然后提取样本集中特征最相似数据的分类标签。如下如图

![KNN-2](/home/fanzone/Documents/ReadingNotes/MachineLearning/KNN-2.jpg)

我们可以从散点图大致推断，这个红点标记可能属于动作片，因为距离那个动作片更近，KNN算法使用距离度量来进行判断。这个电影分类有两个特征值，也就是2维实数的向量空间，可以使用中学学到的两点间距离公式。

![KNN-3](/home/fanzone/Documents/ReadingNotes/MachineLearning/KNN-3.jpg)

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
    print(classCount, "========")
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

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





#### 2.3 示例：手写识别系统



