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
    if classList.count(classList[0]) == len(classList):  # 如果类别完全相同则停止继续划分 
        return classList[0]
    if len(dataSet[0]) == 1 or len(labels) == 0:  # 遍历玩所有特征时返回出现次数最多的类标签
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 选择最优特征
    print(bestFeat)
    print(labels)
    bestFeatLabel = labels[bestFeat]  # 最优特征的标签
    featLabels.append(bestFeatLabel)

    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])  # 删除已使用的特征标签
    featValues = [data[bestFeat] for data in dataSet]  # 得到训练集中所有最优特征的属性值
    uniqueVals = set(featValues)
    for value in uniqueVals:  # 遍历特征，创建决策树
        subLabels = labels[:]
        ds = splitDataSet(dataSet, bestFeat, value)
        myTree[bestFeatLabel][value] = createTree(ds, subLabels, featLabels)
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
- 二是：使用完了所有特征，仍然不能将数据划分仅包含唯一类别的分组，即决策树构建失败，特征不够用，此时说明数据维度不够，由于第二个停止条件无法简单的返回唯一的类标签，这里挑选出现数量最多的类别作为返回值。 （？？？）

#### 3.2 在Python中使用matplotlib绘制树形图

##### 3.2.1 matplotlib注解





##### 3.2.2 构造注解树





#### 3.3 测试和存储分类器







#### 3.4 示例：使用决策树预测隐形眼镜类型



