### 第三章 决策树

决策树(decision tree)是一种基本的分类与回归方法。适用于分类和回归，对应分类树和回归树。

适用数据类型：数值型和标称型。

决策树分类过程：从根节点开始，对实例的某一特征值进行测试，根据测试结果，将实例分配到其子节点，这时，每一个子节点对应着该特征的一个取值。如此递归地对实例进行测试并分类，直至到达叶节点，最后将实例分到叶节点的类中。

生成决策树常用的算法：ID3、C4.5、CART。下面用的ID3算法。

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

在划分数据集之后信息发生的变化称为信息增益。知道如何计算信息增益，我们就可以计算每个特征值划分数据集获得的信息增益。**信息增益最高的特征**就是最好的选择。

**（1）香农熵**

集合信息的度量方式称为香农熵或者熵（entropy）。熵定义为信息的期望值。在信息论与概率统计中，熵表示随机变量不确定性的度量。如果待分类的事物可能划分在多个分类中，则符号xi的信息定义为：

![DT-03](/home/fanzone/Documents/ReadingNotes/MachineLearning/DT-03.png)

其中p(xi)是选择该分类的概率。至于信息为啥会这样定义，记住即可。上式中可以以2为底，也可以以e为底。

通过上式，我们可以得到所有类别的信息，为了计算熵，我们需要计算所有类别所有可能值包含的信息期望值（数学期望），通过下面公式得到：

![](/home/fanzone/Documents/ReadingNotes/MachineLearning/DT-04.png)

其中n是分类的数目。熵越大，随机变量的不确定性越大。

当熵中的概率由数据估计（特别是最大似然估计）得到时，所对应的熵为经验熵。

**问题：什么叫做数据估计？**    ？？？ 

比如10个数据，一个两个类别，A和B类，其中由7个数据属于A类，则该A类的概率为十分之七，其中3个数据属于B类，则B类的概率为十分之三。浅显的解释为，概率是我们根据数据数出来的。我们定义贷款申请样本数据表中的数据为训练数据集D，则训练数据集D的经验熵为H(D)，|D|表示样本容量，及样本个数。设由K个类C<sub>k</sub> =1,2,3...k，|C<sub>k</sub>|为属于类C<sub>k</sub>的样本个数，因此经验熵可以写为：

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

**条件熵：** 条件熵H（Y|X）表示在已知随机变量X的条件下随机变量Y的不确定性。随机变量X给定条件下随机变量Y的条件熵H(Y|X)，定义为X给定条件下Y的条件概率分布的熵对X的数学期望：

![](/home/fanzone/Documents/ReadingNotes/MachineLearning/DT-07.jpg)

其中

![](/home/fanzone/Documents/ReadingNotes/MachineLearning/DT-08.jpg)



同理当条件熵中的概率由数据估计（特别是极大似然估计）得到时，所对应的条件熵称为条件经验熵。

令特征A对训练数据集D的信息增益g(D, A) 定义为集合D的经验熵H(D)与特征A给定条件下D的经验条件熵H（D|A）之差，即

![](/home/fanzone/Documents/ReadingNotes/MachineLearning/DT-09.jpg)

**示例：**

以贷款申请样本为例，令年龄这一列数据为特征A1，一共三种取值，分为青年，中年，老年。看青年的数据，有5个，所以年龄是青年的数据在训练数据集出现的概率为十五分之五，同理中年和老年的数据分别也是三分之一。现在我们只看年龄是青年的数据最终得到贷款的是五分之二（2个是，三个否），同理可得中年和老年最终得到贷款的概率分别为是五分之三，五分之四。所以信息增益为：

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
        value: 需要返回的特征的值
    returns:
        retDataSet: 划分后的数据集
    """
    retDataSet = []
    for featureVec in dataSet:
        if featureVec[axis] == value:
            reducedFeatVec = featureVec[:axis] # 选择第几个特征中值为几的所有数据子集。例如第0个特征的0的数据 splitDataSet(d, 0, 0)
            reducedFeatVec.extend(featureVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """选择最优特征
    Args:
        dataSet: 信息增益最大的特征的索引值
    """
    numFeatures = len(dataSet[0]) - 1  # 特征数量
    baseEntropy = calcShannonEnt(dataSet)  # 计算香农熵
    baseInfoGain = 0.0  # 信息增益
    bestFeature = -1
    for i in range(numFeatures):  # 遍历所有特征
        featList = [data[i] for data in dataSet]
        uniqueVals = set(featList)  # 去重
        newEntropy = 0.0
        for value in uniqueVals:  # 计算信息增益
            subDataSet = splitDataSet(dataSet, i, value)  # 划分子集
            prob = len(subDataSet) / float(len(dataSet))  # 计算子集的概率
            newEntropy += prob * calcShannonEnt(subDataSet)  # 根据公式计算经验条件熵
        infoGain = baseEntropy - newEntropy  # 信息增益
        print(f'第%d个特征的增益为%.3f' % (i, infoGain))
        if infoGain > baseInfoGain:  # 计算信息增益
            baseInfoGain = infoGain  # 更新信息增益，找到最大的信息增益
            bestFeature = i  # 记录信息增益最大的特征的索引值
    return bestFeature  # 返回索引值

if __name__ == "__main__":
    data, labels = createDataSet()
    # x = calcShannonEnt(data)
    # print(x)
    print('最优特征索引值:'+str(chooseBestFeatureToSplit(data)))
"""结果：
第0个特征的增益为0.083
第1个特征的增益为0.324
第2个特征的增益为0.420
第3个特征的增益为0.363
最优特征索引值:2
"""
```





##### 3.1.3 递归构建决策树





#### 3.2 在Python中使用matplotlib绘制树形图

##### 3.2.1 matplotlib注解





##### 3.2.2 构造注解树





#### 3.3 测试和存储分类器







#### 3.4 示例：使用决策树预测隐形眼镜类型



