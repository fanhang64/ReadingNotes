## Softmax回归

虽然叫回归，但是是一个分类问题。

让我们考虑一个简单的图像分类问题，其输入图像的高和宽均为2像素，且色彩为灰度。这样每个像素值都可以用一个标量表示。我 们将图像中的4像素分别记为 `x1,x2,x3,x4`。假设训练数据集中图像的真实标签为狗、猫或鸡（假设可以用4像素表示出这3种动物），这些标签分别对应离散值 `y1,y2,y3` 。
　　我们通常使用离散的数值来表示类别，例如 `y1=1,y2=2,y3=3`。如此，一张图像的标签为 1、2 和 3 这 3 个数值中的一个。虽然我们仍然可以使用回归模型来进行建模，并将预测值就近定点化到 1、2 和 3 这 3 个离散值之一，但这种连续值到离散值的转化通常会 影响到分类质量。因此我们一般使用更加适合离散值输出的模型来解决分类问题。

### Softmax回归模型

Softmax回归跟线性回归一样将输入特征与权重做线性叠加。与线性回归的一个主要不同在于，Softmax 回归的输出值个数等于标签里的类别数。因为一共有4种特征和3种输出动物类别，所以权重包含12个标量（带下标的 w）、偏差包含3个标量（带下标的 b），且对每个输入计算 `o1,o2,o3` 这 3 个输出：

<img src="https://gitee.com/fanhang64/my_images/raw/master/2021/image-20211024205352416.png" alt="image-20211024205352416" style="zoom: 50%;" />

Softmax 回归同线性回归一样，也是一个单层神经网络。由于每个输出 `o1,o2,o3`的计算都要依赖于所有的输入 `x1,x2,x3,x4`, Softmax回归的输出层也是一个全连接层。

<img src="https://gitee.com/fanhang64/my_images/raw/master/2021/image-20211024205833647.png" alt="image-20211024205833647" style="zoom:50%;" />



### Softmax 运算

这⾥要采取的主要⽅法是将模型的输出视作为概率。我们将优化参数以最⼤化观测数据的概率。为了得到
预测结果，我们将设置⼀个阈值，如选择具有最⼤概率的标签。我们希望模型的输出ŷ<sub>j</sub> 可以视为属于类`j `的概率。然后我们可以选择具有最⼤输出值的类别`argmax(j）`，`yj` 作为我们的预测。

既然分类问题需要得到离散的预测输出，一个简单的办法是将输出值`oi` 当作预测类别是 `i` 的置信度, 并将值最大的输出所对应的类作为预测输出，即输出 `argmax(oi)` 。 例如，如果 `o1,o2,o3`分别为 `0.1,10,0.1 `, 由于`o2`最大, 那么预测类别为2，其代表猫。

然而，直接使用输出层的输出有两个问题。一方面，由于输出层的输出值的范围不确定，我们难以直观上判断这些值的意义。例如，刚才举的例子中的输出值 10 表示图像类别为猫，因为该输出值是其他两类的输出值的 `100` 倍。但如果`o1`和`o3`的输出值为10<sup>3</sup>，则之前的输出值`10`却又表示图像类别为猫的概率很低。

另一方面，由于真实标签是离散值，这些离散值与不确定范围的输出值之间的误差难以衡量。

Softmax运算符（softmax operator）解决了以上两个问题。它通过下式将输出值变换成值为正且和为 `1`的概率分布：

<img src="https://gitee.com/fanhang64/my_images/raw/master/2021/image-20211024215722999.png" alt="image-20211024215722999" style="zoom:50%;" />

其中，

<img src="https://gitee.com/fanhang64/my_images/raw/master/2021/image-20211024220132395.png" alt="image-20211024220132395" style="zoom:67%;" />

容易看出 `ŷ1+ŷ2+ŷ3=1`且 `0≤ŷ1,ŷ2,ŷ3≤1`，因此 `ŷ1,ŷ2,ŷ3`是一个合法的概率分布。

这个时候，如果`ŷ2=0.8`，则我们都知道图像类别为猫的概率为`80%`。

**注意：**

- softmax运算不改变预测类别输出。（argmax(oi) = argmax(ŷ)）



### 单样本分类的矢量计算表达式

　为了提高计算效率，我们可以将单样本分类通过矢量计算来表达。在上面的图像分类问题中，假设softmax回归的权重和偏差参数分别为

<img src="https://gitee.com/fanhang64/my_images/raw/master/2021/image-20211024222211333.png" alt="image-20211024222211333" style="zoom: 50%;" />

设高和宽分别为2个像素的图像样本`i`的特征为

<img src="https://gitee.com/fanhang64/my_images/raw/master/2021/image-20211024222243499.png" alt="image-20211024222243499" style="zoom:50%;" />

输出层的输出为

<img src="https://gitee.com/fanhang64/my_images/raw/master/2021/image-20211024222311342.png" alt="image-20211024222311342" style="zoom: 50%;" />

预测为狗、猫或鸡的概率分布为

<img src="https://gitee.com/fanhang64/my_images/raw/master/2021/image-20211024222344280.png" alt="image-20211024222344280" style="zoom:50%;" />

softmax回归对样本 `i` 分类的矢量计算表达式为

<img src="https://gitee.com/fanhang64/my_images/raw/master/2021/image-20211024222415286.png" alt="image-20211024222415286" style="zoom:50%;" />

### 小批量样本分类的矢量计算表达式

　　为了进一步提升计算效率，我们通常对小批量数据做矢量计算。广义上讲，给定一个小批量样本，其批量大小为 `n`，输入个数（特征数）为 `d` ，输出个数（类别数）为 `q`∘ 设批量特征为**X∈R<sup>n×d</sup> **。假设 softmax 回归的权重和偏差参数分别为 **W∈R<sup>d×q</sup>**和 **b∈R<sup>1×q</sup>**。 softmax 回归的矢量计算表达式为:

<img src="https://gitee.com/fanhang64/my_images/raw/master/2021/image-20211024224522236.png" alt="image-20211024224522236" style="zoom:50%;" />

其中的加法运算使用了广播机制，`O`，Y^∈R<sup>n×q</sup> 且这两个矩阵的第 `i` 行分别为样本 `i`的输出 `o(i)`和概率分布 ŷ<sup>(i)</sup>.



**分类 和 回归**

- **回归**估计一个连续值。
  - **单连续**数值的输出
  - 自然区间R
  - 跟真实值的区别作为损失
- **分类**预测一个**离散**类别。  
  - 通常多个输出
  - 输出`i`是预测为第`i`类的置信度

接下来我们看以下如何从回归过度到分类问题。

### 从回归到多类分类

首先对 **类别 ** 进行一位有效编码，假设我们有n个类别，则标号为**y**。假设我们真实的类别为第i个，则`yi`为1，其余元素全部为0。

<img src="https://i.loli.net/2021/10/22/ZspXz2KR1rSkJNW.png" alt="image-20211022155500455" style="zoom:67%;" />

然后可以使用**均方损失**训练，假设我们训练出一个模型，则训练时候，选取`i`，则最大化`Oi`的值为预测结果`ŷ`。

例如：标签`y`将是⼀个三维向量，其中(1; 0; 0)对应于"猫"、(0; 1; 0)对应于"鸡"、(0; 0; 1)对应于"狗"

### 交叉熵损失 (使用交叉熵作为softmax的loss函数)

1. 叉熵损失函数(CrossEntropy Loss)：分类问题中经常使用的一种损失函数

2. 交叉熵能够衡量同一个随机变量中的两个不同概率分布的差异程度，在机器学习中就表示为真实概率分布与预测概率分布之间的差异。**交叉熵的值越小，模型预测效果就越好**。

3. 交叉熵在分类问题中常常与softmax是标配，softmax将输出的结果进行处理，使其多个分类的预测值和为1，再**通过交叉熵来计算损失**。

交叉熵常用来衡量概率的区别。通过下面公式计算交叉熵：

![image-20211026111450793](https://i.loli.net/2021/10/26/d1tVnlKwvH489Q6.png)

将**交叉熵作为损失函数**，如下：

<img src="https://i.loli.net/2021/10/26/Qwqi6IxoNBYOgK1.png" alt="image-20211026112920693" style="zoom:67%;" />

**总结：**

- softmax 回归是一个多类分类模型
- 使用softmax操作子得到每个类的预测置信度
- 使用交叉熵来衡量预测和标号的区别



## 损失函数

我们需要⼀个损失函数来度量预测概率的效果。用来衡量预测值和真实值之间的区别。

### (1) 均方损失 L2 Loss

均方损失为：(1/2) *（预测值- 真实值）^2 

**公式：**   L(y, y') = (1 / 2) * ( y - y')<sup>2</sup> 



###  (2) 绝对值损失函数 L1 Loss 

绝对值损失函数： 预测值 - 真实值的绝对值。

**公式：** L(y, y') = |y - y'|



### (3) huber's Robust Loss

当预测值和真实值的绝对值差的大于1时候，为绝对值误差。

当预测值和真实值的比较接近时候，即绝对值小于1，为均方误差。

**公式：**

![image-20211022160615282](https://i.loli.net/2021/10/22/KCDHGWRp4btZSVY.png)



## 图像分类数据集（如何读取多类分类问题数据集）

**示例：**

```python
%matplotlib inline

import torch
import torchvision
import matplotlib.pyplot as plt

from torch.utils import data
from torchvision import transforms

from d2l import torch as d2l

from IPython import display

d2l.use_svg_display()  # svg 显示图片



# 通过ToTensor() Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor 将图像数据从PIL类型转换到32位浮点数类型
# 并除以255，使得像素数值都在0和1之间

trans = transforms.ToTensor()

# train = True 下载的是训练数据集
# transform = trans 下载下来图片转为tensor。默认为PIL.Image类型
# download = True 从网络下载
mnist_train = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=trans, download=True)


# 测试集，不用于训练，只⽤于评估模型性能
mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans, download=True)

print(len(mnist_train))
print(len(mnist_test))



def get_fashion_minist_labels(labels):
    """返回fashion mnish 数据集的文本标签"""
    
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneakers',
        'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in labels ]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """使用matplotlib 绘制 plot a list of images"""
    
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)  # 隐藏坐标轴
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))  # 大小为18的固定样本
print(X.shape)
print(y.shape)
print(y)

show_images(X.reshape(18,28,28), 2, 9, titles=get_fashion_minist_labels(y))



batch_size = 256
def get_dataloader_workers():
    """使用4个进程来读取数据"""
    
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())

timer = d2l.Timer()

for X, y in train_iter:
    continue

print(f'{timer.stop():.2f} second')


def load_data_fashion_mnist(batch_size, resize=None):
    """下载 Fashion-MNIST 数据集 ，然后加载到内存"""
    
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))  # resize 用于调整图像的大小。将图片变得更大
    trans = transforms.Compose(trans)
    
    mnist_train = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=trans, download=True)

    mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans, download=True)
    
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()),
           data.DataLoader(mnist_test, batch_size, shuffle=True, num_workers=get_dataloader_workers())
           )

```



## Softmax实现

**示例：**

```python
import torch

from d2l import torch as d2l

batch_size = 256

train_iter, test_iter = load_data_fashion_mnist(batch_size)


# 定义W和b
num_inputs = 784  # 对28*28 的图片展平，将它们视为长度为28 * 28 = 784的向量
num_outputs = 10  # 因为我们数据集有10个类别，所以网络输出维度为10

# 图片为长28，宽28
# 对于softmax来说，输入需要为一向量
# 因为我们的数据集有10个类别，所以⽹络输出维度为10。因此，权重将构成⼀个784 x 10的矩阵，偏置将构成⼀个1 x 10的⾏向量。与线性回归⼀样，我们将使
# ⽤正态分布初始化我们的权重W，偏置初始化为0。
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
print(W, b)


# 给定矩阵X，对所有元素求和
X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(X)
print(X.sum(0, keepdim=True))  # keepdim 为2维
print(X.sum(1, keepdim=True))  # 2维


# softmax实现
def softmax(x):
    X_exp = torch.exp(x)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition


# 元整softmax运算
X = torch.normal(0, 1, size=(2,5))
print('X: ', X)
X_prob = softmax(X)
X_prob, X_prob.sum(1)


# 定义模型
def net(X):
    # print(X.shape, "...net...")
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

# 定义损失函数

y = torch.tensor([0, 2])

# y_hat 为两个样本，3个类别的概率
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])

# 拿出0号样本第0个类别的概率。 拿出第1号样本第2个类别的概率。
y_hat[[0, 1], y]

# 实现交叉熵函数
def cross_entropy(y_hat, y):
    # print(y_hat)
    # print(y)
    return -torch.log(y_hat[range(len(y_hat)), y])  # ? 为啥取一段

cross_entropy(y_hat, y)


# 定义准确率函数
def accuracy(y_hat, y):
    """计算预测正确的数量"""
    
    # 判断是否为矩阵
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        
        # 获取每行中的最大元素的下标， 为预测类别
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

accuracy(y_hat, y) / len(y)



# 评估准确率
def evaluate_accuracy(net, data_iter):
    """计算在制定数据集上模型的精度"""
    
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式。 不要计算梯度了，

    metric = Accumulator(2)  # 正确预测数，预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())  # 准确率， 样本数
    # print(metric.data)
    return metric[0] / metric[1]  # 分类正确的样本数 / 总样本数


# 累加器
class Accumulator:
    """n个变量上累加"""
    
    def __init__(self, n):
        self.data = [0.0] * n
    
    def add(self, *args):
        # print(args, ".....accumulator ....")
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0.0] * len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

evaluate_accuracy(net, test_iter)


# 模型训练
def train_epoch_ch3(net, train_iter, loss, updater):  # ch3为第三章
    """训练模型一个迭代周期"""
    
    if isinstance(net, torch.nn.Module):
        net.train()  # 将模型设置为训练模式。 要计算梯度
    
    # 训练损失总和，训练准确度总和，样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度，并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        
        if isinstance(updater, torch.optim.Optimizer):
            # 使用Pytorch内置的优化器和损失函数
            updater.zero_grad()
            l.backward()  # 计算梯度
            updater.step()  # update 参数
            metric.add(float(l) * len(y), accuracy(y_hat, y), y.size().numel())
        else:
            # 使用自己实现的优化器和损失函数
            l.sum().backward()  # 求和，算梯度
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    
    # 返回训练损失和训练准确率
    return metric[0] / metric[2], metric[1] / metric[2]
    
 
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9], legend=['train loss', 'train acc', 'test acc'])
    
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc, ))
    
    train_loss, train_acc = train_metrics
    
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

# 定义Aminator，绘制图像
class Animator:
    """在动画中绘制数据"""
    
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None,\
                 xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-', 'r:'),\
                 nrows=1, ncols=1, index=1, figsize=(3.5, 3.5)):
        # 增量的绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        
        print(nrows, ncols)
        self.fig = plt.figure(figsize=figsize)
        self.axes = plt.subplot(nrows, ncols, 1)
        
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
    
    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)        


lr = 0.1  # 学习率

def updater(batch_size):
    return d2l.sgd([W,b], lr, batch_size)

num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

# 预测
def predict_ch3(net, test_iter, n=6):
    """预测模型"""
    
    for X, y in test_iter:
        break
    trues = get_fashion_minist_labels(y)
    preds = get_fashion_minist_labels(net(X).argmax(axis=1))
    
    titles = [true + "\n" + pred for true, pred in zip(trues, preds)]
    
    show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter, n=7)
```





##  Softmax简易实现

