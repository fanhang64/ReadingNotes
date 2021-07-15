### 第三章 决策树

决策树(decision tree)是一种基本的分类与回归方法。适用于分类和回归，对应分类树和回归树。

适用数据类型：数值型和标称型。

决策树分类过程：从根节点开始，对实例的某一特征值进行测试，根据测试结果，将实例分配到其子节点，这时，每一个子节点对应着该特征的一个取值。如此递归地对实例进行测试并分类，直至到达叶节点，最后将实例分到叶节点的类中。

决策树分割度量方法：信息增益（ID3）、信息增益率（C4.5）、基尼指数（CART）。下面用的ID3算法。

#### 3.1 决策树构造

决策树构建过程包含3个步骤：

1. 特征选择
2. 决策树的生成
3. 决策树剪枝（后面章节说）

**特征选择**在于**选取对训练数据具有分类能力的特征**，也就是决定用哪个特征来划分特征空间。这样可以提高决策树学习的效率。通常特征选择的标准是信息增益（information gain）或信息增益比，以下内容使用信息增益作为选择特征的标准。

##### 3.1.1 信息增益

如下通过所给的训练数据学习一个贷款申请的决策树，用于对未来的贷款申请进行分类，即当新的客户提出贷款申请时，根据申请人的特征利用决策树决定是否批准贷款申请。

![DT-01](/home/fanzone/Documents/ReadingNotes/MachineLearning/DT-01.jpg)

我们可以通过上面这个表，得到如下可能的决策树（不止这两个）,分别由两个不同特征的根节点构成。

![DT-02](/home/fanzone/Documents/ReadingNotes/MachineLearning/DT-02.jpg)

如图，图(a)的根节点为**年龄**，有3个取值（青年，中年，老年）。图(b)根节点的特征为**工作**，对应2个取值。

**问题：选择哪种特征作为根节点更好？**

如果一个特征能够更好的分类，按照这个特征将训练数据分割成子集，使的各个子集在当前条件下有最好的分类，那么就更应该选择这个特征。信息增益就能很好的表达这一直观的准则。

**问题：什么是信息增益？**

在划分数据集之后信息发生的变化称为信息增益。如果知道如何计算信息增益，我们就可以计算每个特征值划分数据集获得的信息增益。**信息增益最高的 特征**就是最好的选择。

**（1）香农熵**

集合信息的度量方式称为香农熵或者熵（entropy）。熵定义为信息的期望值。在信息论与概率统计中，熵表示随机变量不确定性的度量。如果待分类的事物可能划分在多个分类中，则符号xi的信息定义为：

![DT-03](/home/fanzone/Documents/ReadingNotes/MachineLearning/DT-03.png)

其中p(xi)是选择该分类的概率。至于信息为啥会这样定义，记住即可。上式中可以以2为底，也可以以e为底。

通过上式，我们可以得到所有类别的信息，为了计算熵，我们需要计算所有类别所有可能值包含的信息期望值（数学期望），通过下面公式得到：

![](/home/fanzone/Documents/ReadingNotes/MachineLearning/DT-04.png)

其中n是分类的数目。当熵中的概率由数据估计（特别是最大似然估计）得到时，所对应的熵为**经验熵**。

我们定义贷款申请样本数据表中的数据为训练数据集D，则训练数据集D的**经验熵为H(D)**，|D|表示样本容量，及样本个数。设由K个类C<sub>k</sub> =1,2,3...k，|C<sub>k</sub>|为属于类C<sub>k</sub>的样本个数，因此经验熵可以写为：

![](/home/fanzone/Documents/ReadingNotes/MachineLearning/DT-05.png)

根据此公式计算经验熵H(D)，分析贷款申请样本数据表中的数据。最终分类结果只有两类，即放贷和不放贷。根据表中的数据统计可知，在15个数据中，9个数据结果为放贷，6个数据的结果为不放贷，所以数据集D的经验熵H(D)为

![](/home/fanzone/Documents/ReadingNotes/MachineLearning/DT-06.jpg)

由此可知数据集D的经验熵H(D)的值为0.971。

**（2）编写代码计算经验熵**

在编写代码之前，我们先对数据集进行属性标注。

- 年龄： 0代表青年，1代表中年，2代表老年
- 有工作：0代表否，1代表是
- 有自己房子：0代表否，1代表是
- 信贷情况：0代表一般，1代表好，2代表非常好
- 类别（是否给贷款）：no代表否，yes代表是。

```python
import math
def createDataSet():
    dataSet = [
        [0, 0, 0, 0, 'no'],
        [0, 0, 0, 1, 'no'],
        [0, 1, 0, 1, 'yes'],
        [0, 1, 1, 0, 'yes'],
        [0, 0, 0, 0, 'no'],
        [1, 0, 0, 0, 'no'],
        [1, 0, 0, 1, 'no'],
        [1, 1, 1, 1, 'yes'],
        [1, 0, 1, 2, 'yes'],
        [1, 0, 1, 2, 'yes'],
        [2, 0, 1, 2, 'yes'],
        [2, 0, 1, 1, 'yes'],
        [2, 1, 0, 1, 'yes'],
        [2, 1, 0, 2, 'yes'],
        [2, 0, 0, 0, 'no'],
    ]
    labels = ['不放贷', '放贷']  # 分类属性
    return dataSet, labels

def calcShannonEnt(dataSet):
    """计算给定数据集的经验熵

    Args:
        dataSet： 给定数据集
    Returns:
        shannonEnt: 经验熵（香农熵）
    """
    numEntries = len(dataSet)
    labelsCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]  # 提取标签信息
        if currentLabel not in labelsCounts.keys():
            labelsCounts[currentLabel] = 0
        labelsCounts[currentLabel] += 1  # 计数
    shanonEnt = 0.0
    for key in labelsCounts:
        prob = float(labelsCounts[key]) / numEntries  # 选择这个标签的概率 no的概率 0.4 yes的概率0.6
        shanonEnt -= prob * math.log(prob, 2)  # H(D) = -（pi*log(pi) + pj*log(pj)） = -pi*log(pi) - pj*log(pj）
    return shanonEnt

if __name__ == "__main__":
    data, labels = createDataSet()
    x = calcShannonEnt(data)
    print(x) # 0.9709505944546686
```

**(3) 信息增益**

信息增益是相对于特征而言的，信息增益越大，特征对最终的分类结果影响也就越大，我们就应该选择对分类结果影响最大的那个特征作为我们的分类特征。

**条件熵：** 条件熵H（Y|X）表示在已知随机变量X的条件下随机变量Y的不确定性（？？？）。随机变量X给定条件下随机变量Y的条件熵H(Y|X)，定义为X给定条件下Y的条件概率分布的熵对X的数学期望：

![](/home/fanzone/Documents/ReadingNotes/MachineLearning/DT-07.jpg)

其中

![](/home/fanzone/Documents/ReadingNotes/MachineLearning/DT-08.jpg)



同理当条件熵中的概率由数据估计（特别是极大似然估计）得到时，所对应的条件熵称为**经验条件熵**。

令特征A对训练数据集D的信息增益g(D, A) 定义为集合D的经验熵H(D)与特征A给定条件下D的经验条件熵H（D|A）之差，即**公式：** `信息增益 = 分割前的经验熵（信息熵） - 分割后的经验条件熵H(D|A)`。

![](/home/fanzone/Documents/ReadingNotes/MachineLearning/DT-09.jpg)

**注意：**

- 分割后的子节点的信息熵越小，表示分类结果约纯，信息增益就越大。

**示例：**

以贷款申请样本为例，令年龄这一列数据为特征A1，一共三种取值，分为青年，中年，老年。

此时的经验熵（即分割前）`H(D)=-9/15 * log(9/15) - 6/15 * log(6/15)`

看青年，中年，老年的数据，都各有5个，共有15个数据，各占`5/15`。现在我们只看年龄是青年的数据最终得到贷款的是五分之二（2个是，三个否）。即`H(D|A1) = 1/3 * H(D1)+ 1/3 * H(D2) + 1/3 * H(D3)`

所以信息增益为：

![](/home/fanzone/Documents/ReadingNotes/MachineLearning/DT-10.jpg)

这里的D1，D2，D3分别为数据集D中A1（年龄）取值为青年，中年，老年的样本子集。

计算A2，A3，A4特征的信息增益如下：

![](/home/fanzone/Documents/ReadingNotes/MachineLearning/DT-13.jpg)



![](/home/fanzone/Documents/ReadingNotes/MachineLearning/DT-11.jpg)



![](/home/fanzone/Documents/ReadingNotes/MachineLearning/DT-12.jpg)

通过比较信息增益，由于A3（有房子）的信息增益最大，所以选择A3作为最优特征。

##### 3.1.2 划分数据集

`splitDataSet`函数是用来选择各个特征的子集的，比如选择年龄（第0个特征值）的青年，可以通过`splitDataSet(dataSet, 0, 0)`来返回的子集为青年的5个数据集。

```python
import math
def createDataSet():
    dataSet = [
        [0, 0, 0, 0, 'no'],
        [0, 0, 0, 1, 'no'],
        [0, 1, 0, 1, 'yes'],
        [0, 1, 1, 0, 'yes'],
        [0, 0, 0, 0, 'no'],
        [1, 0, 0, 0, 'no'],
        [1, 0, 0, 1, 'no'],
        [1, 1, 1, 1, 'yes'],
        [1, 0, 1, 2, 'yes'],
        [1, 0, 1, 2, 'yes'],
        [2, 0, 1, 2, 'yes'],
        [2, 0, 1, 1, 'yes'],
        [2, 1, 0, 1, 'yes'],
        [2, 1, 0, 2, 'yes'],
        [2, 0, 0, 0, 'no'],
    ]
    labels = ['不放贷', '放贷']  # 分类属性
    return dataSet, labels

def calcShannonEnt(dataSet):
    """计算给定数据集的经验熵
    Args:
        dataSet： 给定数据集
    Returns:
        shannonEnt: 经验熵（香农熵）
    """
    numEntries = len(dataSet)
    labelsCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]  # 提取标签信息
        if currentLabel not in labelsCounts.keys():
            labelsCounts[currentLabel] = 0
        labelsCounts[currentLabel] += 1  # 计数
    shanonEnt = 0.0
    for key in labelsCounts:
        prob = float(labelsCounts[key]) / numEntries  # 选择这个标签的概率 no的概率 0.4 yes的概率0.6
        shanonEnt -= prob * math.log(prob, 2)  # H(D) = -（pi*log(pi) + pj*log(pj)） = -pi*log(pi) - pj*log(pj）
    return shanonEnt

def splitDataSet(dataSet, axis, value):
    """按照给定特征划分数据集
    Args:
        dataSet: 待划分数据集
        axis: 选取的划分数据集的特征（某一列）
        value: 选取特征取为对应的值
    returns:
        retDataSet: 划分后的数据集
    """
    retDataSet = []
    for featureVec in dataSet:
        if featureVec[axis] == value:
            reducedFeatVec = featureVec[:axis]  # 选择第几个特征中值为几的所有数据子集。例如第0个特征的0的数据 splitDataSet(d, 0, 0)
            reducedFeatVec.extend(featureVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """选择最优特征
    Args:
        dataSet: 信息增益最大的特征的索引值
    """
    numFeatures = len(dataSet[0]) - 1  # 特征数量，选取第一行，计算特征值个数
    baseEntropy = calcShannonEnt(dataSet)  # 计算香农熵
    baseInfoGain = 0.0  # 信息增益
    bestFeature = -1  # 最优特征值索引下标
    for i in range(numFeatures):  # 遍历所有特征, 第0列为第一个特征即i=0
        featList = [data[i] for data in dataSet]
        uniqueVals = set(featList)  # 去重, 为某个特征的所有取值，例如第0列为[0，1,2]
        newEntropy = 0.0  # 用于计算条件经验熵,即H(D|A)
        for value in uniqueVals:  # 计算信息增益
            subDataSet = splitDataSet(dataSet, i, value)  # 划分子集, 第0列特征，按取值为0，1，2时候划分数据
            prob = len(subDataSet) / float(len(dataSet))  # 计算子集的概率

            newEntropy += prob * calcShannonEnt(subDataSet)  # 根据公式计算经验条件熵
        infoGain = baseEntropy - newEntropy  # 信息增益
        print(f'第%d个特征的增益为%.3f' % (i, infoGain))
        if infoGain > baseInfoGain:  # 选择最大信息增益的索引下标
            baseInfoGain = infoGain  # 更新信息增益，找到最大的信息增益
            bestFeature = i  # 记录信息增益最大的特征的索引值
    return bestFeature  # 返回索引值

if __name__ == "__main__":
    data, labels = createDataSet()
    x = calcShannonEnt(data)  # 计算经验熵
    # print(x)
    # r = splitDataSet(data, 0, 1)  # 选取第0列特征值为1的数据（不包括特征值第0列）
    # print(r)
    print('最优特征索引值:'+str(chooseBestFeatureToSplit(data)))

"""结果：
第0个特征的增益为0.083
第1个特征的增益为0.324
第2个特征的增益为0.420
第3个特征的增益为0.363
最优特征索引值:2
"""
```

**总结：**

上述工作原理是通过原始数据集，基于最好的特征划分数据集，由于特征值可能多于两个，因此可能存在大于两个分支的数据集划分。第一次划分之后，数据集被向下传递到树的分支的下一个结点。在这个结点上，我们可以再次划分数据。因此我们可以采用递归的原则处理数据集。

##### 3.1.3 递归构建决策树

**构建决策树方法：** 从根节点（root node）开始，对节点计算所有可能的特征的信息增益，选择信息增益最大的特征作为节点的特征，**由该特征的不同取值建立子节点**；在对子节点递归的调用上述方法，构建决策树；直到所有特征的信息增益均很小或没有特征可以选择为止，最后得到一个决策树。

继续分析开始的图表，由于特征A3（是否有房子）的信息增益最大，故选取特征A3作为根节点的特征。它将训练集D划分为两个子集D1（是）和D2（否），**由于D1只有同一类的样本点**（在最后一列是否给贷款的值都为`是`），故D1成为一个叶节点，节点的类标记为`是`。

而 D2则需要特征A1（年龄），A2（有工作）和A4（信贷情况）中选择新的特征，计算各个特征的信息增益：

![](/home/fanzone/Documents/ReadingNotes/MachineLearning/DT-14.jpg)

故选择A2（是否有工作）作为新的子节点，由于A2有两个取值（是，否），当`是`包含3个样本，属于同一类，所以为叶子节点；当`否`包含6个节点，也属于同一类，所以也为叶子节点。从而生成一个决策树，如下：

![](/home/fanzone/Documents/ReadingNotes/MachineLearning/DT-15.jpg)



**使用字典存储决策树：**

```python
# 这里使用字典存储决策树
{
    '有房子'：{
        0：{
            '有工作':{
                0: 'no',
                1: 'yes'
            }
        }，
        1： 'yes'
    }
}
```

**构造决策树代码实现：**

```python
import math
import numpy as np
import operator
def createDataSet():
    dataSet = [
        [0, 0, 0, 0, 'no'],
        [0, 0, 0, 1, 'no'],
        [0, 1, 0, 1, 'yes'],
        [0, 1, 1, 0, 'yes'],
        [0, 0, 0, 0, 'no'],
        [1, 0, 0, 0, 'no'],
        [1, 0, 0, 1, 'no'],
        [1, 1, 1, 1, 'yes'],
        [1, 0, 1, 2, 'yes'],
        [1, 0, 1, 2, 'yes'],
        [2, 0, 1, 2, 'yes'],
        [2, 0, 1, 1, 'yes'],
        [2, 1, 0, 1, 'yes'],
        [2, 1, 0, 2, 'yes'],
        [2, 0, 0, 0, 'no'],
    ]
    # labels = ['不放贷', '放贷']  # 分类属性
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']  # 特征标签（分类属性）
    return dataSet, labels

def majorityCnt(classList):
    """统计classList中出现次数最多的元素
    Args:
        classList: 类标签列表

    Returns:
        sortedClassCount[0][0]: 出现次数最多的元素（类标签）
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

def createTree(dataSet, labels, featLabels):
    """构造决策树

    Args:
        dataSet: 训练数据集
        labels: 分类属性标签
        featLabels: 存储选择的最优特征标签

    Returns:
        myTree: 构造的决策树
    """
    classList = [data[-1] for data in dataSet]  # 分类标签（是否贷款yes no）
    if classList.count(classList[0]) == len(classList):  # 统计类别个数，如果类别个数与列表相同则停止继续划分，即结果为同一类
        return classList[0]
    if len(dataSet[0]) == 1 or len(labels) == 0:  # 这里如果使用完了所有特征,仍不能将数据划分仅包含唯一类别分组，即特征不够。这时遍历玩所有特征时返回出现次数最多的类标签【取值0 还是 1】返回
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 选择最优特征索引下标
    print(bestFeat)
    print(labels)
    bestFeatLabel = labels[bestFeat]  # 最优特征的标签
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])  # 删除已使用的特征标签
    featValues = [data[bestFeat] for data in dataSet]  # 得到训练集中所有最优特征的属性值(取值)
    uniqueVals = set(featValues)
    print(uniqueVals)  # {0, 1}
    for value in uniqueVals:  # 遍历特征，创建决策树
        subLabels = labels[:]
        ds = splitDataSet(dataSet, bestFeat, value) # 划分数据
        myTree[bestFeatLabel][value] = createTree(ds, subLabels, featLabels)  # 划分后数据继续执行递归
    return myTree

if __name__ == "__main__":
    data, labels = createDataSet()
    x = calcShannonEnt(data)  # 计算经验熵
    # print(x)
    # r = splitDataSet(data, 0, 1)  # 选取第0列特征值为1的数据（不包括特征值第0列）
    # print(r)
    # print('最优特征索引值:'+str(chooseBestFeatureToSplit(data)))
    featLabels = []
    myTree = createTree(data, labels, featLabels)
    print(myTree)
“”“
{'有自己的房子': {0: {'有工作': {0: 'no', 1: 'yes'}}, 1: 'yes'}}
”“”
```

递归创建决策树时，递归有两个中止条件：

- 一是：所有的类标签完全相同，则直接返回该类标签
- 二是：使用完了所有特征，仍然不能将数据划分仅包含唯一类别的分组，即决策树构建失败，特征不够用，此时说明数据维度不够，由于第二个停止条件无法简单的返回唯一的类标签，这里挑选出现数量最多的类别作为返回值。

#### 3.2 在Python中使用matplotlib绘制树形图

##### 3.2.1 matplotlib注解

下面通过`matplotlib`模块绘制决策树的树形图，其中`matplotlib`模块有一个注解工具`annotations`，可以在图形中添加文本注释，注解通常用于解释数据的内容。

```python
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def getNumLeaves(myTree):
    """获取决策树的叶子节点数

    Args:
        myTree: 决策树
    returns:
        numLeaves: 叶子节点数
    """
    numLeaves = 0
    firstStr = next(iter(myTree))  # 遍历dict的key  例如a = {name:{'zs': {}, 'ls':{}}} 则next(iter(a)) => name
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(key) == 'dict':
            numLeaves += getNumLeaves(secondDict[key])
        else:
            numLeaves += 1
    return numLeaves

def getNumDepth(myTree):
    """ 获取决策树层数

    Args:
        myTree: 决策树
    returns:
        numDepth: 决策树层数
    """
    maxDepth = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(key)  == 'dict':
            thisDepth = 1 + getNumDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle="<-")  # 定义箭头格式
    font = FontProperties(fname=r"/usr/share/fonts/simsun.ttc", size=10)  # 设置中文字体
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction', #  绘制结点
        xytext=centerPt, textcoords='axes fraction',
        va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, FontProperties=font)


def plotMidText(cntrPt, parentPt, txtString):
    """标注有向表属性值

    Args:
        cntrPt: 用于计算标注位置
        parentPt: 用于计算标注位置
        txtString: 标注的内容
    """
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]  # 计算标注位置
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
    """绘制决策树

    Args:
        myTree: 决策树
        parentPt: 标注的内容
        nodeTxt: 节点名
    """
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")  # 设置结点格式
    leafNode = dict(boxstyle="round4", fc="0.8")  # 设置叶结点格式
    numLeafs = getNumLeaves(myTree)  # 获取决策树叶结点数目，决定了树的宽度
    depth = getNumDepth(myTree)  # 获取决策树层数
    firstStr = next(iter(myTree))
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)  # 中心位置
    plotMidText(cntrPt, parentPt, nodeTxt)  # 标注有向边属性值
    plotNode(firstStr, cntrPt, parentPt, decisionNode)  # 绘制结点
    secondDict = myTree[firstStr]  # 下一个字典，也就是继续绘制子结点
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD  # y偏移
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':  # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            plotTree(secondDict[key],cntrPt,str(key))  # 不是叶结点，递归调用继续绘制
        else:  # 如果是叶结点，绘制叶结点，并标注有向边属性值   
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


def createPlot(inTree):
    """创建绘制面板

    Args:
        inTree: 决策树
    """
    fig = plt.figure(1, facecolor='white')  # 创建fig
    fig.clf()  # 清空fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # 去掉x、y轴
    plotTree.totalW = float(getNumLeaves(inTree))  # 获取决策树叶结点数目
    plotTree.totalD = float(getNumDepth(inTree))  # 获取决策树层数
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0  # x偏移
    plotTree(inTree, (0.5,1.0), '')  # 绘制决策树
    plt.show()  # 显示绘制结果 
```



#### 3.3 测试和存储分类器

##### 3.3.1 测试分类器

在执行数据分类时，需要决策树以及用于构造树的标签向量，然后程序比较测试数据与决策树上的数值，递归执行该过程直到进入叶子节点，最后将测试数据定义为叶子节点所属的类型。

```python
import numpy as np
import math

def createDataSet():
    dataSet = [
        [0, 0, 0, 0, 'no'],
        [0, 0, 0, 1, 'no'],
        [0, 1, 0, 1, 'yes'],
        [0, 1, 1, 0, 'yes'],
        [0, 0, 0, 0, 'no'],
        [1, 0, 0, 0, 'no'],
        [1, 0, 0, 1, 'no'],
        [1, 1, 1, 1, 'yes'],
        [1, 0, 1, 2, 'yes'],
        [1, 0, 1, 2, 'yes'],
        [2, 0, 1, 2, 'yes'],
        [2, 0, 1, 1, 'yes'],
        [2, 1, 0, 1, 'yes'],
        [2, 1, 0, 2, 'yes'],
        [2, 0, 0, 0, 'no'],
    ]
    # labels = ['不放贷', '放贷']  # 分类属性
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']  # 特征标签（分类属性）
    return dataSet, labels

def calcShannonEnt(dataSet):
    """计算给定数据集的经验熵
    Args:
        dataSet： 给定数据集
    Returns:
        shannonEnt: 经验熵（香农熵）
    """
    numEntries = len(dataSet)
    labelsCounts = {}

    for featVec in dataSet:
        currentLabel = featVec[-1]  # 提取标签信息
        if currentLabel not in labelsCounts.keys():
            labelsCounts[currentLabel] = 0
        labelsCounts[currentLabel] += 1  # 计数
    shanonEnt = 0.0
    for key in labelsCounts:
        prob = float(labelsCounts[key]) / numEntries  # 选择这个标签的概率 no的概率 0.4 yes的概率0.6
        shanonEnt -= prob * math.log(prob, 2)  # H(D) = -（pi*log(pi) + pj*log(pj)） = -pi*log(pi) - pj*log(pj）
    return shanonEnt

def splitDataSet(dataSet, axis, value):
    """按照给定特征划分数据集
    Args:
        dataSet: 待划分数据集
        axis: 选取的划分数据集的特征（某一列）
        value: 选取特征取为对应的值
    returns:
        retDataSet: 划分后的数据集
    """
    retDataSet = []
    for featureVec in dataSet:
        if featureVec[axis] == value:
            reducedFeatVec = featureVec[:axis]  # 选择第几个特征中值为几的所有数据子集。例如第0个特征的0的数据 splitDataSet(d, 0, 0)
            reducedFeatVec.extend(featureVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """选择最优特征
    Args:
        dataSet: 信息增益最大的特征的索引值
    """
    numFeatures = len(dataSet[0]) - 1  # 特征数量，选取第一行，计算特征值个数
    baseEntropy = calcShannonEnt(dataSet)  # 计算香农熵
    baseInfoGain = 0.0  # 信息增益
    bestFeature = -1  # 最优特征值索引下标
    for i in range(numFeatures):  # 遍历所有特征, 第0列为第一个特征即i=0
        featList = [data[i] for data in dataSet]
        uniqueVals = set(featList)  # 去重, 为某个特征的所有取值，例如第0列为[0，1,2]
        newEntropy = 0.0  # 用于计算条件经验熵,即H(D|A)
        for value in uniqueVals:  # 计算信息增益
            subDataSet = splitDataSet(dataSet, i, value)  # 划分子集, 第0列特征，按取值为0，1，2时候划分数据
            prob = len(subDataSet) / float(len(dataSet))  # 计算子集的概率
            newEntropy += prob * calcShannonEnt(subDataSet)  # 根据公式计算经验条件熵
        infoGain = baseEntropy - newEntropy  # 信息增益
        print(f'第%d个特征的增益为%.3f' % (i, infoGain))
        if infoGain > baseInfoGain:  # 选择最大信息增益的索引下标
            baseInfoGain = infoGain  # 更新信息增益，找到最大的信息增益
            bestFeature = i  # 记录信息增益最大的特征的索引值
    return bestFeature  # 返回索引值

def majorityCnt(classList):
    """统计classList中出现次数最多的元素
    Args:
        classList: 类标签列表

    Returns:
        sortedClassCount[0][0]: 出现次数最多的元素（类标签）
    """
    classCount = {}

    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

def createTree(dataSet, labels, featLabels):
    """构造决策树
    Args:
        dataSet: 训练数据集
        labels: 分类属性标签
        featLabels: 存储选择的最优特征标签
    Returns:
        myTree: 构造的决策树
    """
    classList = [data[-1] for data in dataSet]  # 分类标签（是否贷款yes no）
    if classList.count(classList[0]) == len(classList):  # 统计类别个数，如果类别个数与列表相同则停止继续划分，即结果为同一类
        return classList[0]
    if len(dataSet[0]) == 1 or len(labels) == 0:  # 这里如果使用完了所有特征,仍不能将数据划分仅包含唯一类别分组，即特征不够。这时遍历玩所有特征时返回出现次数最多的类标签【取值0 还是 1】返回
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 选择最优特征索引下标
    print(bestFeat)
    print(labels)
    bestFeatLabel = labels[bestFeat]  # 最优特征的标签
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])  # 删除已使用的特征标签
    featValues = [data[bestFeat] for data in dataSet]  # 得到训练集中所有最优特征的属性值(取值)
    uniqueVals = set(featValues)
    print(uniqueVals)  # {0, 1}
    for value in uniqueVals:  # 遍历特征，创建决策树
        subLabels = labels[:]
        ds = splitDataSet(dataSet, bestFeat, value) # 划分数据
        myTree[bestFeatLabel][value] = createTree(ds, subLabels, featLabels)  # 划分后数据继续执行递归
    return myTree

def classify(tree, feat_labels, test_vec):
    """使用决策树分类
    Args:
        tree: 已经生成的决策树
        feat_labels: 存储选择的最优特征标签
        test_vec: 测试数据列表，顺序对应最优特征标签
    """
    first_key = next(iter(tree))
    second_dict = tree[first_key]
    feat_index = feat_labels.index(first_key)
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if isinstance(second_dict[key], dict):
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]
    return class_label

if __name__ == '__main__':
    data_set, labels = createDataSet()
    feat_labels = []
    my_tree = createTree(data_set, labels, feat_labels)
    print(my_tree)

    ret = classify(my_tree, feat_labels, [0, 1])  # test_vec: [0, 1]  # 表示无房子，有工作
    print(ret)
"""
yes
"""
```

##### 3.3.2 决策树的存储

使用python自带的序列化模块`pickle`

```python
import pickle

def save_tree(in_tree, fname):
    with open(fname, 'wb') as fp:
        pickle.dump(in_tree, fp)

def load_tree(fname):
    in_tree = None
    with open(fname, 'rb') as fp:
        in_tree = pickle.load(fp)
    return in_tree
```



#### 3.4 示例：使用决策树预测隐形眼镜类型

sklearn.tree模块提供了决策树模型，用于解决分类和回归问题。

docs:  [http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

```python
from sklearn.tree import DecisionTreeClassifier
DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, max_features=None, ...)

"""
criterion： 特征选择标准，默认gini 基尼不纯度, 可以设置为entropy 香农熵。
splitter: 特征划分的标准，可选参数， 默认best,可以设置为random。best是根据算法选择最佳的切分特征。
max_features: 划分时考虑的最大特征数，可选，默认为None。
max_depth: 决策树最大深度，树的层数，默认为None, 此时决策树建立的时候不会限制子树的深度。
...
"""

# methods
decision_path(X[, check_input])  # 返回决策树中的决策路径
fit(X, y, [sample_weight, check_input, …])  # 根据训练集构建决策树分类器
get_depth()  # 返回决策树的深度
get_n_leaves()  # 返回决策树的叶子数
predict(X, [check_input])  # 预测X的类别或回归值
score(X, y, [sample_weight])  # 返回给定测试数据和标签的平均精确度
```

**示例：**

```python
from io import StringIO

import pydotplus
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.preprocessing import LabelEncoder


if __name__ == "__main__":
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    print(lenses)
    lenses_target = []
    for x in lenses:
        lenses_target.append(x[-1])
    print(lenses_target)
    lenses_labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    clf = DecisionTreeClassifier(max_depth=4)

    # 这里需要对数据进行编码，
    # 因为fit（）函数不能接受string类型的数据， 对数据编码需要先生成pandas数据

    # lenses = clf.fit(lenses, y=lenses_labels)
    # print(lenses)

    lenses_list = []
    for x in lenses:
        _d = {}
        for k, v in zip(lenses_labels, x):
            _d[k] = v
        lenses_list.append(_d)
    print(lenses_list)
    lenses_pd = pd.DataFrame(lenses_list)
    print(lenses_pd)
    
    le = LabelEncoder()
    for col in lenses_pd.columns:
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
        print(lenses_pd[col])
    print(lenses_pd)

    clf = clf.fit(lenses_pd.values, lenses_target)
    dot_data = StringIO()

    export_graphviz(clf, out_file=dot_data, feature_names=lenses_labels, \
        class_names=clf.classes_, filled=True, rounded=True, special_characters=True)
    
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf('tree.pdf')
	
    # 预测
    r = clf.predict([[1,1,1,0]])
    print(r)
"""
hard
"""
```

此时生成一个pdf文件,查看可视化的效果图。

**总结：**

- 优点：
  - 易于理解和解释，决策树可以可视化。
  - 几乎不需要数据预处理。其他经常需要数据标准化，删除缺失值等。
  - 使用树的花费（例如预测数据）是训练数据点数量的对数。
  - 可以同时处理数值变量和分类变量。其他大都适用于分析一种变量的集合。
  - 可以处理多值输出变量问题。
  - 使用白盒模型。
- 缺点：
  - 决策树可能创建一个过于复杂的树，并不能很好的预测数据。也就是过拟合，需要设置一个叶子节点需要的最小样本数量，或者数的最大深度，可以避免过拟合。
