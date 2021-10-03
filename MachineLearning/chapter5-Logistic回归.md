## Logistic 逻辑回归

 逻辑回归是一种**分类**算法，通常用于解决**二分类**的问题，用来表示某件事情发生的可能性，任务是尽可能的拟合决策边界。逻辑分类主要思想是根据现有的数据对分类边界线建立回归公式，以此进行分类。

主要应用：预测明天下雨的可能性（下雨，不下雨），预测银行卡欺诈的可能性（欺诈，不是欺诈），购买一件商品的可能性（买，不买）等。

**注意：**

- 二分类问题可以拓展到多分类问题



### 5.1 基于Logistic回归和Sigmoid函数的分类



假设现在有一些数据点，我们用一条直线对这些点进行拟合，这条直线称为最佳拟合直线，这个拟合的过程叫做回归。

进而可以得到这些点的拟合直线方程，我们如何根据这个回归方程进行分类？



**二值型输出分类函数**

我们想要的函数是：能够接受所有的输入然后预测出类别。 

例如：在两个类的情况下，上面的回归方程输出0或1。这类函数称为海维塞得阶跃函数，或者称为单位越界函数。然而这个阶跃函数的问题在于，函数在跳跃点从0瞬间跳跃到1，这个瞬间跳跃过程有时很难处理。

类似的函数也有类似的性质（可以输出0或1），在数学上，更容易处理，这就是sigmoid函数，公式如下：

![image-20211003133641856](/home/fanzone/.config/Typora/typora-user-images/image-20211003133641856.png)

下图给出了sigmoid函数在不同坐标尺度下的两条曲线图。当x为0.5时，sigmoid函数值为0.5。随着x的增大，对应的sigmoid也随之逼近1；随着x的减小，sigmoid值也将逼近于0。如果横坐标足够大，则sigmoid函数看起来像一个阶跃函数。

<img src="/home/fanzone/.config/Typora/typora-user-images/image-20211003133912082.png" alt="image-20211003133912082" style="zoom:50%;" />

因此，为了实现logistic回归分类器，我们可以在每个特征上乘以一个**回归系数**，然后将所有的结果值相加，将这个总和带入sigmoid函数中，进而得到一个范围在0-1之间的数值。

任何概率大于0.5的都被分为1类，小于0.5的被分为0类。所以logistic一种概率估计。比如：这里sigmoid函数得到的值为0.5，可以理解为给定数据和参数，数据被分入1类的概率为0.5。



### 5.2 基于最优化方法的最佳回归系数确定

sigmoid函数的输入记为z，由下面公式得到：

```mathematica
z = w0x0 + w1x1 + w2x2 + ... + wnxn
```

如果采用向量的写法，上述公式可以写成，z = w<sup>T</sup>x，它表示将两个数值向量对应元素相乘然后相加得到`z`。其中x为分类器的输入数据，向量w也就是我们要找到的最佳参数（系数），从而使用分类器尽可能的精确。

为了寻找最佳参数，需要用到的最优化理论的一些知识，这里使用的梯度上升法（gradient ascent）。

#### 5.2.1 梯度上升法

**数学知识：**

- 方向导数

  某一方向的变化率

- 梯度

  > `fx(x0, y0)i + fy(x0, y0)j`，这个**向量**称为函数f(x, y)在点`P(x0, y0)`的梯度，记做`gradf(x0, y0)`。即`gradf(x0, y0) = f'x(x0, y0)i + f'y(x0, y0)j`。

**总结：**

```python
向量 = 值 + 方向
梯度= 向量
梯度 = 梯度值 + 梯度方向
```

**梯度上升法的思想**

首先看一个简单求极大值的例子，例如：`f(x) = -x^2 + 4x` ，其图像是凸的抛物线，如果求极值，要先求导数，`f'(x) = -2x + 4` ，令导数为0，可得f(x)的极大值为f(2) = 4。

但是在实际情况下，函数不会像这一简单，计算求出函数的导数，可能也很难计算函数的极值。这时候我们就可以用迭代的方法来做。一点一点的逼近极值。这种寻找最佳拟合参数的方法，就是最优化算法。这个逼近过程用数学公式表达为：

```
xi+1 = xi + α*(∂f(xi) / xi)  ， 其中α为移动步长，也就是学习速率，控制更新的幅度。注意：移动步长会影响最终结果的拟合程度，最好的方法就是随着迭代次数更改移动步长。
```

**示例：**

```python
def gradient_ascent_test():
    """梯度上升测试 - 函数求导

        求函数f(x) = -x^2 + 4x 导数
    """
    def derivative_func(x):  # f(x)的导函数
        return -2 * x + 4
    
    x_old = -1  # 初始值，随机的小于x_new的值
    x_new = 0  # 从（0，0）开始
    alpha = 0.01  # 步长，也就是学习速率，控制更新的幅度
    presission = 0.00000001  # 精度，更新阈值（临界值）
    while abs(x_new - x_old) > presission:
        print(f"x_old:{x_old}, x_new:{x_new}")
        x_old = x_new
        x_new = x_old + alpha * derivative_func(x_old)
    print(x_new)  # 近似值  1.999999515279857

if __name__ == "__main__":
    gradient_ascent_test()
```

结果很近似真实极值为2，这个过程，就是梯度上升算法。

**注意：**

```python
梯度上升和梯度下降？
答：本质相同。区别在于叫误差函数还是叫目标函数。
如果目标函数是损失函数，那就是最小化损失函数，来求函数的最小值，就用梯度下降。
如果目标函数是似然函数，就是最大化似然函数，来求函数的最大值，就是梯度上升。
逻辑回归中，损失函数和似然函数互为正负关系。
```

#### 5.2.2 训练算法： 使用梯度上升找到最佳参数

**梯度上升法伪代码：**

```
每个回归系数初始化为1
重复R次：
	计算整个数据集的梯度
	使用alpha * gradient 更新回归系数向量
	返回回归系数
```

**梯度上升法具体实现**

```python
import numpy as np 
import matplotlib.pyplot as plt

def load_data_set():
    """加载数据集
        一行数据： -0.017612	14.053064	0
        第一列为x轴上值，第二列看为y轴的值，第三列为分类标签
    """
    data_mat = []
    label_mat = []
    fr = open('./testSet.txt')
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])  # string类型，需要转成float类型
        label_mat.append(int(line_arr[-1]))
    fr.close()
    return data_mat, label_mat

def plot_data_set():
    """绘制数据集

    """
    data_mat, label_mat = load_data_set()
    data_arr = np.array(data_mat)
    n = np.shape(data_mat)[0]  # 数据个数

    x_cord1 = []
    y_cord1 = []
    x_cord2 = []
    y_cord2 = []
    for i in range(n):
        if label_mat[i] == 1:  # 1样本 正样本
            x_cord1.append(data_arr[i,1])
            y_cord1.append(data_arr[i,2])
        else:
            x_cord2.append(data_arr[i,1])  # 0为负样本
            y_cord2.append(data_arr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_cord1, y_cord1, s=20, c='red', marker='s', alpha=0.5)  # s为marker的size,c为marker的颜色
    ax.scatter(x_cord2, y_cord2, s=20, c='green', alpha=0.5)  # marker 参数默认为圆形，marker=s为方块
    plt.title('data set')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == "__main__":
    plot_data_set()
```

<img src="/home/fanzone/.config/Typora/typora-user-images/image-20211003173630301.png" alt="image-20211003173630301" style="zoom:50%;" />

从上图可以看出，假设sigmoid函数的输入记为z，那么`z=w0x0+w1x1+w2x2`，即可将数据分隔割开，其中，x0为全是1的向量，x1位数据集的第一列数据，x2为数据集的第二列数据。

另外z=0，则`0 = w0*1+w1x1+w2x2`，横坐标为x1，纵坐标为x2，这个方程的未知参数为w0，w1，w2为我们需要求的回归参数（最优参数）。



#### 5.2.3 分析数据：画出决策边界





#### 5.2.4 训练算法：随机梯度上升



### 5.3 示例： 从疝气病症预测病马的死亡率

