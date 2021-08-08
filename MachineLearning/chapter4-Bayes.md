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

示例：

```python
import pandas as pd

def load_datasets():
    posting_list = [  # 切分的词条
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cut', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthkless', 'dog', 'food', 'stupid']
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

`train_mat`是所有的词条向量组成的列表，它里面存放的是根据`my_vocab_list`向量化的词条向量，下面编写训练函数，使用词条向量计算概率

```python
```







#### 4.5 示例：使用朴素贝过滤垃圾邮件





#### 4.6 示例：使用朴素贝叶斯分类器从个人广告中获取区域倾向