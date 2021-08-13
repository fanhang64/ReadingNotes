### 第四章 朴素贝叶斯

朴素贝叶斯是有监督学习算法，解决的分类的问题，如客户是否流失，是否值得投资，信用登记评定，文档分类等多分类问题，下面将给出一些使用概率论分类的方法。

朴素：只做原始，最简单的假设，所有特征之前是统计独立的。

假设某个样本有`a1，a2，a3，a4，.... an`个属性，则`P(X) = P(a1,a2,a3,...an) = P(a1) * P(a2) * ...P(an)`

#### 4.1 基于贝叶斯决策理论的分类方法

朴素贝叶斯是贝叶斯理论的一部分。

假设我们有一个数据集，由两类数据组成，数据分布如下：

![Bayes-01](/home/fanzone/Documents/ReadingNotes/MachineLearning/Bayes-01.jpg)

现在用p1(x,y) 表示数据点(x,y)属于类别1（红色原点）的概率，用p2(x, y)表示数据点(x, y)属于类别2（蓝色三角形）的概率，那么对于一个新的数据点(x, y)，可以通过下面的规则来判断它的类别：

- 如果p1(x, y) > p2(x, y) 则属于类别1
- 如果p1(x, y) < p2(x, y) 则属于类别2

这里会选择概率较高对应的类别，这就是贝叶斯决策核心理论，即选择具有最高概率的决策。

#### 4.2 条件概率

条件概率指在事件B发生的情况下，事件A发生的概率，用`P(A|B)`来表示。

若只有A,B两个事件，那么`P(A|B) = P(AB) / P(B)`。

由乘法公式可得： `P(AB) = P(A)P(B|A) 或 P(AB) = P(B)P(A|B)`

可得条件概率公式：`P(A|B) = P(A)P(B|A) / P(B)`

#### 4.3 贝叶斯推断

由条件概率公式得，

**先验概率：** 即`P(A)`，即在B发生之前，对A事件概率的有个判断。

**后验概率：** 即`P(A|B)`，即在B发生条件下，对A的概率评估。

**可能性函数：** 即`P(B|A) / P(B) `这是一个调整因子，使得预估概率接近真实概率。

因此条件概率可以理解为下面的式子：

```
后验概率 = 先验概率 * 调整因子

P(A|B) = P(A) * (P(B/A) / P(B))
```

上述为贝叶斯推断的含义，先预估一个**先验概率**，然后加上实验结果，看这个实验到底是增强了还是削弱了先验概率，因此得到更接近事实的**后验概率**。

如果`可能性函数即调整因子 > 1`，则意味着**先验概率**被增强，事件A发生的概率可能性变大；

如果`可能性函数 = 1`，意味着事件B对判断事件A的可能性无帮助；

如果`可能性函数 < 1`，意味着**先验概率**被削弱，事件A的可能性变小。



**朴素贝叶斯分类的原理**

**思想基础：**对于给出的`待分类项`，求解<u>在此项出现的条件下，各个类别出现的概率</u>，哪个最大，就认为此待分类项属于哪个类别。

**定义如下：**

1. 设`x={a1, a2, a3, ..., am}` 为一个待分类项，而a为x的一系列特征属性。
2. 有类别分类`C = {y1, y2, y3, ...., yn}`
3.  分别计算`P(y1| x), P(y2| x), P(y3|x), ..., P(yn|x)`
4. 如果`P(yk|x) = max{P(y1|x), P(y2|x), P(y3|x), ..., P(yn|x)}，则待分类项x属于yk类别。`

**注意： 如何计算P(yi| x), 其中i为1...n**

1. 找到`已知分类`的待分类项集合，这个集合叫做`训练样本集`。

2. 统计得到在各个类别条件下各个特征属性的概率，即

   `P(a1|y1), P(a2|y1), P(a3|y1), P(a4|y1), ...P(am|y1)；P(a1|y2), P(a2|y2), P(a3|y2), ..., P(am|y2)；... P(a1|yn), P(a2|yn), P(a3|yn), ..., P(am|yn)； `

3. 根据贝叶斯定理可得

   `P(yi| x) = P(x|yi) * P(yi) / P(x)`

4. 又因特征属性是条件独立的，故可得

   `P(x|yi)P(yi) = P(a1|yi) * P(a2|yi) * P(a3|yi) * ... P(am|yi) * P(yi) = P(yi) π P(aj|yi) 其中j为1...m`



**示例：**  门诊病人情况表

| 症状   | 职业   | 疾病   |
| ------ | ------ | ------ |
| 打喷嚏 | 护士   | 感冒   |
| 打喷嚏 | 农夫   | 过敏   |
| 头痛   | 建筑工 | 南震荡 |
| 头痛   | 建筑工 | 感冒   |
| 打喷嚏 | 教师   | 感冒   |
| 头痛   | 教师   | 脑震荡 |

问：现在又来了一个病人，是**打喷嚏的建筑工**，他感冒的概率多大?

根据贝叶斯公式：

```
P(A|B) = P(AB) / P(B) = P(A)P(B|A) / P(B)
```

则可得：

```python
P(感冒| 打喷嚏 * 建筑工) = P(感冒) * P(打喷嚏 * 建筑工 | 感冒) / P(打喷嚏 * 建筑工)

P(感冒) = 1 / 2 
# 由于A和B独立，可得P(AB|C) = P(A|C)*P(B|C)。
P(打喷嚏 * 建筑工 | 感冒) = P(打喷嚏 | 感冒)* P(建筑工 | 感冒)
P(打喷嚏 | 感冒) = 2 / 3 
P(建筑工 | 感冒) = 1 / 3
# 由于A和B独立 <=> P(AB)=P(A)*P(B)
P(打喷嚏 * 建筑工) = P(打喷嚏) * P(建筑工) = 1 / 2 * 2 / 6 
P(感冒| 打喷嚏 * 建筑工) = P(感冒) * P(打喷嚏 | 感冒) * P(建筑工 | 感冒) / P(打喷嚏 * 建筑工) = (1/2 * 2 / 3 * 1 / 3) / (1 / 2 * 2 / 6 ) = 2 / 3
```

#### 4.4 使用Python进行文本分类

以在线社区留言为例，构建一个快速过滤器，如果某条留言使用了负面或者侮辱性的语言，那么就将该留言标志为不当内容。对此问题，建立两个类别：侮辱类和非侮辱类，用1和0分别表示。

我们把文本看成单词向量或者词条向量，也就啥将句子转换为向量。首先考虑文档中出现的所有文档中的单词，在决定将那些单词纳入词汇表或者所有的词汇集合，然后必须将每一篇文档转换为词汇表的向量。 

**开发流程：**

```
收集数据
准备数据: 从文本中构建词向量
分析数据: 检查词条确保解析的正确性
训练算法: 根据构建后词向量计算概率
测试算法
使用算法
```

**示例：**

```python
import pandas as pd

def load_datasets():
    posting_list = [  # 切分的词条
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cut', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    class_vec = [0,1,0,1,0,1]  # 类别标签向量，1代表侮辱类，0代表非侮辱类
    return posting_list, class_vec

def create_vocab_list(data_sets):
    """创建一个包含所有文档中出现的不重复词的列表。

    Args:
        data_sets: 整理的样本数据集
    returns:
        vocab_set: 返回的不重复的词条列表，也就是词汇表（就是所有单词放在一个列表中，去重操作）
    """
    vocab_set = set()

    for d in data_sets:
        vocab_set = vocab_set | set(d)  # 取并集
    return list(vocab_set)

def set_of_words_to_vec(vocab_list, input_dataset):
    """ 根据vocablist词汇表，将input dataset量化,向量的每个元素为1或0

    Args:
        vocab_list : 不重复词汇列表
        input_set : 切分的词条列表
    returns: 
        return_vec : 返回将词条向量化后列表。（转为1和0的列表）
    """
    return_vec = [0] * len(vocab_list)
    for word in input_dataset:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print("the word: %s is not in my vocabulary!" % word)
    return return_vec

if __name__ == "__main__":
    data_sets, classes_vec = load_datasets()  # data_sets是原始的词条列表
    print(data_sets, classes_vec)
    my_vocab_list = create_vocab_list(data_sets)  # my_vocab_list是词汇表，用于将词条向量化，如果一个单词在词汇表中，那么相应位置记作为1,否则为0
    print(my_vocab_list)
    train_mat = []
    for data_set in data_sets:
        r = set_of_words_to_vec(my_vocab_list, data_set)
        train_mat.append(r)
    print(train_mat)
```

`train_mat`是所有的词条向量组成的列表，它里面存放的是根据`my_vocab_list`向量化的词条向量，下面编写训练函数，通过之前的条件概率公式，对每个类别计算概率，然后比较概率的大小。

```python
P(Ci| W) = P(Ci) * P(W|Ci) / P(W)  # 其中i表示类别，w表示一个向量，由多个值组成，一个向量中数值个数和词汇表词个数相同。
```

**训练函数:**

```python
def _trainNB0(train_mat, train_category):
    """ 朴素贝叶斯分类器训练函数
    Args:
        train_mat : 训练文档（一条文档就是一个留言，一个list）矩阵，即set_of_words_to_vec 返回的词条向量化后的列表
        train_category : 训练类别标签向量。
    returns:
        p0_vect: 非侮辱性的条件概率数组  [P(W0|C0), P(W1|C0), P(W2|C0), P(W3|C0), ...]
        p1_vect: 侮辱性的条件概率数组    [P(W0|C1), P(W1|C1), P(W2|C1), P(W3|C1), ...]
        p_abusive 文档属于侮辱类的概率    P(C1)
    """
    num_train_docs = len(train_mat)  # 统计文档的数目
    num_words = len(train_mat[0])  # 统计每篇文档的词条数   32 
    p_absive = sum(train_category) / float(num_train_docs)  # 计算类别为1即为侮辱类的概率P(Ci) 
    p0_num = np.zeros(num_words)  # 构造单词出现列表
    p1_num = np.zeros(num_words) 

    p0_denom = 0.0  # 整个数据集单词出现的总数
    p1_denom = 0.0 
    for i in range(num_train_docs):  # range(6)  [0 1 0 1 0  1] 

        if train_category[i] == 1:  # 统计侮辱类的条件概率所需的数据
            p1_num += train_mat[i]  # 将侮辱类文档向量相加,即[0,1,1,....] + [0,1,1,....] + ... ->[0,2,3,...]
            p1_denom += sum(train_mat[i])  # 统计侮辱单词出现的总数
        else:   # 统计非侮辱类的条件概率所需的数据
            p0_num += train_mat[i]
            p0_denom += sum(train_mat[i])

    p0_vect = p0_num / p0_denom  # 统计非侮辱类数组，每个单词出现的概率，即每个单词占非侮辱类总单词数比率, [1,2,3,5]/90->[1/90,...]                               
    p1_vect = p1_num / p1_denom
    return p0_vect, p1_vect, p_absive

def classify(vec_classify, p0_vect, p1_vect, p_class1):
    """分类函数

    Args:
        vec_classify : 待分类向量化后一条文档（留言）   [0, 1, 1, 0, 0, 1]
        p0_vect : 是非侮辱类每个单词概率数组
        p1_vect : 是侮辱类每个单词概率数组
        p_class1 : 文档属于侮辱类概率
    """ 
    #  vec_classify * p1_vect  将每个词与对应的概率相关联???
    p1 = reduce(lambda x,y: x*y, vec_classify*p1_vect) * p_class1  # P(W|C1) * P(C1)
    p0 = reduce(lambda x,y: x*y, vec_classify*p0_vect) * (1.0 - p_class1)  # P(W|C0) * P(C0) ，这里没有除以P(W), 因为两个式子，P(W)相同，只考虑分子。
    print("p0:", p0)
    print("p1:", p1)
    
    if p1 > p0:
        return 1
    else:
        return 0

def testing():
    data_sets, classes_vec = load_datasets()
    my_vocab_list = create_vocab_list(data_sets)
    train_mat = []
    for data_set in data_sets:
        r = set_of_words_to_vec(my_vocab_list, data_set)
        train_mat.append(r)
    p0_v,p1_v,p_ab = trainNB0(train_mat,classes_vec)

    test_ = ['love', 'my', 'dalmation']
    test_doc = np.array(set_of_words_to_vec(my_vocab_list, test_))
    if classify(test_doc, p0_v, p1_v, p_ab) == 1:
        print(test_, '属于侮辱类')
    else:
        print(test_, "属于非侮辱类")
    test_ = ['stupid', 'garbage']
    test_doc = set_of_words_to_vec(my_vocab_list, test_)
    if classify(test_doc, p0_v, p1_v, p_ab) == 1:
        print(test_, "属于侮辱类")
    else:
        print(test_, "属于非侮辱类")
"""
p0: 0.0
p1: 0.0
['love', 'my', 'dalmation'] 属于非侮辱类
p0: 0.0
p1: 0.0
['stupid', 'garbage'] 属于非侮辱类
"""
```

**训练函数出现的问题:**

1. 在利用贝叶斯分类器对文档进行分类时，要计算多个概率的乘积，以获得文档属于哪个类别的概率，即计算`P(W1|1) * P(W2|1) * P(W3|1)`， 如果其中一个概率为0，则最终结果为0。
2. 存在`下溢出`的问题，由于在python中很多非常小的数字相乘，使得最后得不到正确的结果。

**解决办法：**

1. 将所有**词的出现次数初始化为1，将分母初始化为2**。（拉普拉斯平滑）

2. 解决下溢出问题，对乘积取自然对数**log**。下图给出了函数`f(x)`与` ln(f(x)) `的曲线。可以看出，它们在相同区域内同时增加或者减少，并且在相同点上取到极值。它们的取值虽然不同，但不影响最终结果。 

   <img src="/home/fanzone/Documents/ReadingNotes/MachineLearning/Bayes-02.jpg" alt="Bayes-02" style="zoom:80%;" />

**示例：**

```python
def trainNB0(train_mat, train_category):
    """模型训练函数
    Args:
        train_mat : 词条向量化后列表
        train_category : 类别
    """
    num_train_docs = len(train_mat)  # 总文档数
    num_words = len(train_mat[0])  # 总单词数
    p_absive = sum(train_category) / float(num_train_docs)  # 侮辱性文档出现的概率
    p0_num = np.ones(num_words)  # 单词出现的次数列表，非侮辱类单词出现次数列表
    p1_num = np.ones(num_words)

    p0_denom = 2.0  # 统计非侮辱类单词的总数，初始化为2，拉普拉斯平滑
    p1_denom = 2.0

    for x in range(num_train_docs):
        if train_category[x] == 1:
            p1_num += train_mat[x]  # 侮辱类单词次数相加
            p1_denom += sum(train_mat[x])  # 统计侮辱类所有文档的总单词数
        else:
            p0_num += train_mat[x]
            p0_denom += sum(train_mat[x])
    p0_vect = np.log(p0_num / p0_denom)  # 以e为底取log
    p1_vect = np.log(p1_num / p1_denom)
    return p0_vect, p1_vect, p_absive

def classifyNB(vec_classify, p0_vect, p1_vect, p_class1):
    """分类函数
       将之前分类函数的乘法转换为加法：
           乘法： P(Ci|W0W1W2...Wn) = P(W0W1W2...Wn | Ci) * P(Ci) / P(W0W1W2...Wn) = P(W0|Ci) * P(W1|Ci) * P(W2|Ci)...P(Wn | Ci) * P(Ci) / P(W0W1W2...Wn)
           优化后加法： P(Ci|W0W1W2...Wn) = P(W0W1W2...Wn | Ci) * P(Ci) / P(W0W1W2...Wn) -> P(W0|Ci)*P(W1|Ci)....P(Wn|Ci)P(Ci) ->（取log）-> log(P(W0|Ci)) + log(P(W1|Ci)) + ... + log(P(Wn|Ci)) 
                      + log(P(Ci))
    Args:
        vec_classify: 待分类的向量 
        p0_vect: 非侮辱类单词概率数组，即 [log(P(W0|C0)), log(P(W1|C0)), log(P(W2|C0)), log(P(W3|C0)), ...]
        p1_vect: 侮辱类单词概率数组，即  [log(P(W0|C1)), log(P(W1|C1)), log(P(W2|C1)), log(P(W3|C1)), ...]
        p_class1: 文档属于类别1侮辱类概率
    """
    p1 = sum(vec_classify * p1_vect) + np.log(p_class1)  # 取log，以e为底，计算P(W|C1) * P(C1)
    p0 = sum(vec_classify * p0_vect) + np.log(1.0 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0

def testing():
    data_sets, classes_vec = load_datasets()
    my_vocab_list = create_vocab_list(data_sets)
    train_mat = []
    for data_set in data_sets:
        r = set_of_words_to_vec(my_vocab_list, data_set)
        train_mat.append(r)
    # p0_v,p1_v,p_ab = _trainNB0(train_mat,classes_vec)
    p0_v,p1_v,p_ab = trainNB0(train_mat,classes_vec)

    test_ = ['love', 'my', 'dalmation']
    test_doc = np.array(set_of_words_to_vec(my_vocab_list, test_))
    if classifyNB(test_doc, p0_v, p1_v, p_ab) == 1:
        print(test_, '属于侮辱类')
    else:
        print(test_, "属于非侮辱类...")

    test_ = ['stupid', 'garbage']
    test_doc = set_of_words_to_vec(my_vocab_list, test_)
    if classifyNB(test_doc, p0_v, p1_v, p_ab) == 1:
        print(test_, "属于侮辱类")
    else:
        print(test_, "属于非侮辱类")
```

#### 4.5 示例：使用朴素贝过滤垃圾邮件

**步骤：**

```
收集数据
准备数据: 将文本文件解析成词条向量
分析数据: 检测词条确保解析的正确性
训练算法: 使用我们之前建立的trainNB0()函数
测试算法: 使用classifyNB()，并构建一个新的测试函数来计算文档集的错误率。
使用算法
```

提供有两个文件夹`ham`和`spam`，`spam`文件下的`txt`文件为垃圾邮件。

**示例： 文件内容解析为向量**

```python
import re
import numpy as np

def text_prase(big_string):
    """接收一个字符串并将其解析为字符串列表
    Args:
        big_string : 输入字符串
    """
    list_of_tokens = re.split(r"\W+", big_string)  # 使用正则表达式来切分句子，其中分隔符是除单词、数字外的任意字符串
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]  # 过滤少于2个字符的字符串

def create_vocab_list(data_sets):
    """创建词汇表，即将所有文档中单词整理成不重复的单词列表。
    Args:
        data_sets: 数据集
    """
    vocab_set = set()
    for document in data_sets:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)

if __name__ == "__main__":
    doc_list = []
    class_list = []
    
    for file_index in range(1, 26):
        word_list = text_prase(open("spam/%d.txt" % file_index).read())
        doc_list.append(word_list)
        class_list.append(1)  # 1为垃圾邮件
        word_list = text_prase(open("ham/%d.txt" % file_index).read())
        doc_list.append(word_list)
        class_list.append(0)  # 0为非垃圾邮件
    my_vocab_list = create_vocab_list(doc_list)
    print(my_vocab_list)
```

下面将数据集分为训练集和测试集，使用`留存交叉验证`的方法来测试朴素贝叶斯分类器的准确性。

```python
def bag_of_words_to_vec(vocab_list, input_set):
    """根据词汇表，将input_set向量化，构建词袋模型

    Args:
        vocab_list : 词汇表
        input_set : 输入的文档
    """
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1  # 与词集模型区别就是每个词出现多次就累加。
    return return_vec

def trainNB0(train_mat, train_category):  # 训练函数
    num_train_docs = len(train_mat) 
    num_words = len(train_mat[0])                            
    p_abusive = sum(train_category) / float(num_train_docs)
    p0_num = np.ones(num_words)
    p1_num = np.ones(num_words)
    p0_denom = 2.0
    p1_denom = 2.0
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_mat[i]
            p1_denom += sum(train_mat[i])
        else:
            p0_num += train_mat[i]
            p0_denom += sum(train_mat[i])

    p1Vect = np.log(p1_num / p1_denom)                             
    p0Vect = np.log(p0_num / p0_denom)

    return p0Vect, p1Vect, p_abusive   

def classifyNB(vec_classify, p0_vect, p1_vect, p_class1):  # 分类函数
    p1 = sum(vec_classify * p1_vect) + np.log(p_class1)
    p0 = sum(vec_classify * p0_vect) + np.log(1.0 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0

def spam_test():
    """
        测试分类器
    """
    doc_list = []  # 文档列表
    class_list = []  # 类别列表
    for i in range(1, 26):  # 解析文件到list
        word_list = text_prase(open(f"spam/{i}.txt").read())
        doc_list.append(word_list)
        class_list.append(1)  # 1为 垃圾邮件
        word_list = text_prase(open(f"ham/{i}.txt").read())
        doc_list.append(word_list)
        class_list.append(0)  # 0代表非垃圾邮件
    my_vocab_list = create_vocab_list(doc_list)  # 创建词汇表
    # 构建训练集（40）和测试集（10）
    train_set_index = list(range(50))
    test_set_index = []
    for i in range(10):
        rand_index = int(random.uniform(0, len(train_set_index)))  # random.uniform  均匀分布返回[a - b)的值,float类型]
        test_set_index.append(train_set_index[rand_index])
        del train_set_index[rand_index]  # 在训练集中移除
    # 向量化
    train_mat = []
    train_classes = []
    for doc_index in train_set_index:  # 遍历索引下标
        v = bag_of_words_to_vec(my_vocab_list, doc_list[doc_index])
        train_mat.append(v)
        train_classes.append(class_list[doc_index])
    # 训练
    p0_vect, p1_vect, p1_classes = trainNB0(train_mat, train_classes)  # p0_vect 为[P(W0|C0), P(W1|C0), P(W2|C0), P(W3|C0), ...]
    # 分类
    error_count = 0
    for doc_index in test_set_index:
        word_vector = bag_of_words_to_vec(my_vocab_list, doc_list[doc_index])
        if classifyNB(word_vector, p0_vect, p1_vect, p1_classes) != class_list[doc_index]:
            error_count += 1
            print("分类错误的测试集：",doc_list[doc_index])
    print('错误率：%.2f%%' % (float(error_count) / len(test_set_index) * 100))
"""
分类错误的测试集： ['home', 'based', 'business', 'opportunity', 'knocking', 'your', 'door', 'don', 'rude', 'and', 'let', 'this', 'chance', 'you', 'can', 'earn', 'great', 'income', 'and', 'find', 'your', 'financial', 'life', 'transformed', 'learn', 'more', 'here', 'your', 'success', 'work', 'from', 'home', 'finder', 'experts']
错误率：10.00%
"""
```

#### 4.6 示例：使用朴素贝叶斯分类器-新浪新闻分类（sklearn）

##### 4.6.1 中文分词模块jieba

**安装：**

```
pip install jieba
```

**示例： 读取文件，切分中文语句**

```python
import os
import jieba

def text_parse(folder_path):
    """文件解析
    Args:
        folder_path : 路径
    """
    folder_list = os.listdir(folder_path)
    data_list = []
    class_list = []
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path, folder)
        files = os.listdir(new_folder_path)

        sample_count = 1  # 每类样本数最多100个（财经100样本，体育100样本...）
        for file in files:
            if sample_count > 100:
                break
            with open(os.path.join(new_folder_path, file)) as fp:
                raw = fp.read()
            word_cut = jieba.cut(raw)  # 中文分词, 返回迭代器
            word_list = list(word_cut)

            data_list.append(word_list)
            class_list.append(folder)
            sample_count += 1
    print(data_list)
    print(class_list)

if __name__ == "__main__":
    text_parse("./Sample")
```

##### 4.6.2 文本特征提取

```python
import os
import jieba
import random

def text_parse(folder_path, test_size=0.2):
    """文本处理
    Args:
        folder_path : 路径
        test_size (float, optional): 测试集占比. Defaults to 0.2.
    """
    folder_list = os.listdir(folder_path)
    data_list = []  # [[], [], [], [], ...]
    class_list = []  # [xxx, xxx, xxx, ...]
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path, folder)
        files = os.listdir(new_folder_path)
        sample_count = 1  # 每类样本数最多100个,（财经100样本，体育100样本...）
        for file in files:
            if sample_count > 100:
                break
            with open(os.path.join(new_folder_path, file)) as fp:
                raw = fp.read()
            word_cut = jieba.cut(raw)  # 中文分词, 返回迭代器
            word_list = list(word_cut)
            data_list.append(word_list)
            class_list.append(folder)
            sample_count += 1
    # 划分数据集
    data_class_list = list(zip(data_list, class_list))  #  [([1, 1, 1], 'a'), ([2, 2, 2], 'b'), ([3, 3, 3], 'c')]
    random.shuffle(data_class_list)  # 顺序打乱
    index = int(len(data_list) * test_size) + 1
    train_list = data_class_list[index:]  # 训练集 [11:]
    test_list = data_class_list[:index]  # 测试集
    train_data_list, train_class_list = zip(*train_list)  # [[1,1,1], [2,2,2]]
    test_data_list, test_class_list = zip(*test_list)  # ['a', 'b', 'c']
    # 统计词频
    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict:
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1
    # 根据频次排序
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda x:x[1], reverse=True)  # 递减
    all_words_list, _ = zip(*all_words_tuple_list)
    all_words_list = list(all_words_list)  # tuple to list
    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list

def make_words_set(words_file):
    """读取文件
    Args:
        words_file : 文件路径
    Returns:
        word_set: 文件内容去重
    """
    word_set = set()
    with open(words_file) as fp:
        for line in fp.readlines():
            word = line.strip()
            if len(word) > 0:
                word_set.add(word)
    return word_set

def words_dict(all_words_list, deleteN, stopwords_set:set):
    """ 文本特征提取

    Args:
        all_words_list : 训练集 所有的文本列表
        deleteN : 删除词频最高的N个词
        stopwords_set (set): 用于过滤的词集合
    """
    feature_words = []
    n = 1
    for t in range(deleteN, len(all_words_list)):
        if n > 1000:  # feature_words 的维度不能大于1000
            break
        # 如果这个词不是数字，并且不在过滤列表中，并且单词长度大于1小于5，那么这个词就可以作为特征词
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
        n += 1
    return feature_words

if __name__ == "__main__":
    folder_path = "./Sample"
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = text_parse(folder_path)
    print(all_words_list)
    stopwords_file = './stopwords_cn.txt'
    stopwords_set = make_words_set(stopwords_file)
    feature_words = words_dict(all_words_list, 100, stopwords_set)
    print(feature_words)
```

##### 4.6.3 通过sklearn构建朴素贝叶斯分类器

在`scikit-learn`中，一共有3个素朴贝叶斯的分类算法，分别为高斯分布的朴素贝叶斯`GaussianNB`， 多项式分布的朴素贝叶斯`MultinomialNB`，伯努利分布的朴素贝叶斯`BernoulliNB`，

```python
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB

def text_features(train_data_list, test_data_list, feature_words):
    """根据feature_words将文本向量化

    Args:
        train_data_list : 训练集
        test_data_list :  测试集
        feature_words : 特征集
    Returns:
        train_feature_list: 训练集向量化列表
        test_feature_list: 测试集向量化列表
    """
    def _text_features(text, feature_words):
        text_words = set(text)
        return [1 if word in text_words else 0 for word in feature_words]
    train_feature_list = [_text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [_text_features(text, feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list

def text_classifier(train_feature_list, test_feature_list, train_class_list, test_class_list):
    """新闻分类器

    Args:
        train_feature_list (list): 训练集向量化后的特征文本
        test_feature_list (list): 测试集向量化的特征文本
        train_class_list (list): 训练集分类标签
        test_class_list (list): 测试集分类标签
    """
    classifer = MultinomialNB()
    classifer = classifer.fit(train_feature_list, train_class_list)
    test_accuracy = classifer.score(test_feature_list, test_class_list)
    return test_accuracy

if __name__ == "__main__":
    folder_path = "./Sample"
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = text_parse(folder_path)
    print(all_words_list)
    stopwords_file = './stopwords_cn.txt'
    stopwords_set = make_words_set(stopwords_file)
    test_accuracy_list = []
    deleteNs = range(0, 1000, 20)
    for deleteN in deleteNs:
        feature_words = words_dict(all_words_list, deleteN, stopwords_set)
        train_feature_list, test_feature_list = text_features(train_data_list, test_data_list, feature_words)
        test_accuracy = text_classifier(train_feature_list, test_feature_list, train_class_list, test_class_list)
        test_accuracy_list.append(test_accuracy)
    plt.figure()
    plt.plot(deleteNs, test_accuracy_list)
    plt.title("Relationship of deleteNs and test_accuracy...")
    plt.xlabel('deleteNs')
    plt.ylabel('test_accuracy')
    plt.show()
```



**总结：**

1. 在训练朴素贝叶斯分类器之前，需要处理训练集。
2. 要根据提取的分类特征将文本向量化，才能训练素朴贝叶斯分类器。
3. 拉普拉斯平滑对于改善贝叶斯分类器的分类有积极的作用。

参考：

- 《机器学习实战》
- 《统计学习方法》
- https://www.cnblogs.com/sxron/p/5452821.html
- https://cuijiahua.com/blog/2017/11/ml_4_bayes_1.html
