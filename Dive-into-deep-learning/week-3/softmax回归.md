## Softmax回归

虽然叫回归，但是是一个分类问题。



**分类 和 回归**

- **回归**估计一个连续值。
  - **单**连续数值的输出
  - 自然区间R
  - 跟真实值的区别作为损失
- **分类**预测一个离散类别。  
  - 通常多个输出
  - 输出i是预测为第i类的置信度



### 分回归到多类分类 - 均方损失

对类别进行一位有效编码

<img src="https://i.loli.net/2021/10/22/ZspXz2KR1rSkJNW.png" alt="image-20211022155500455" style="zoom:67%;" />









### Softmax和 交叉熵损失







**总结：**

- softmax 回归是一个多类分类模型
- 使用softmax操作子得到每个类的预测置信度
- 使用交叉熵来衡量预测和标号的区别



## 损失函数

用来衡量预测值和真实值之间的区别。

### (1) 均方损失 L2 Loss

**公式：**   L(y, y') = (1 / 2) * ( y - y')<sup>2</sup> 



###  (2) 绝对值损失函数 L1 Loss 

**公式：** L(y, y') = |y - y'|



### (3) huber's Robust Loss

**公式：**

![image-20211022160615282](https://i.loli.net/2021/10/22/KCDHGWRp4btZSVY.png)



## 图像分类数据集（如何读取多类分类问题数据集）







## Softmax实现

公式：

<img src="https://i.loli.net/2021/10/24/iwVcDfJoHZhB546.png" alt="image-20211024173002388" style="zoom:67%;" />





##  Softmax简易实现

