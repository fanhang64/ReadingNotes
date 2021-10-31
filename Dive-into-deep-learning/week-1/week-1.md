### 安装需要的包
pip install -y jupyter d2l torch torchvision

pip install rise  # 可选，将jupyter页面变为幻灯片形式

### 下载代码并执行
wget https://zh-v2.d2l.ai/d2l-zh.zip
unzip d2l-zh.zip
jupyter notebook



### 矩阵求导

**标量（scalar)**

> **一个标量表示一个单独的数**，它不同于线性代数中研究的其他大部分对象（通常是多个数的数组）。我们用斜体表示标量。标量通常被赋予**小写的变量名称**。如：a

 **向量（vector）**

> **一个向量表示一组有序排列的数**。通过次序中的索引，我们可以确定每个单独的数。通常我们赋予**向量粗体的小写变量名称**。当我们需要明确表示向量中的元素时，我们会将元素排列成一个方括号包围的纵柱：**如 \*a\*.**
>
> ![image-20211009155622079](https://i.loli.net/2021/10/09/olMXHNhIybw2fJ6.png)

**矩阵（matrix）**

> 矩阵是一个二维数组，其中每一个元素由两个索引所确定。一个有m行，n列，每个元素都属于 RR 的矩阵记作 A∈Rm×n. **通常使用大写变量名称**，如A
>
>  

**张量（tensor）**

> **超过两维的数组叫做张量**。
>
> 在某些情况下，我们会讨论坐标超过两维的数组，一般的，一**个数组中的元素分布在若干维坐标的规则网格中，我们称之为张量**。我们使用字体 A 来表示张量“A”。张量A中坐标为(i,j,k) 的元素记作 Ai,j,k .





在高等数学中，我们已经学过了标量对标量的求导，比如标量y和x的求导，可以表示为`∂ y / ∂x`。

有时候，我们会有一组y<sub>i</sub> ，i=1,2,3,4...m来对一个标量x的求导，那么会得到一组标量的求导结果，即`∂y<sub>i</sub>  / ∂x，其中i为1,2，...m`。

如果我们把这组标量写成向量的形式，即得到维度m的一个向量y对一个标量的x求导，那么结果也是有一个m维的向量`∂ y / ∂x`。

可见所谓的**向量对标量的求导**，其实就是向量里的每个分量分别对标量求导，然后把结果按照向量表示而已。

#### 2. 矩阵向量求导定义

根据求导的自变量和因变量是标量，向量还是矩阵。我们有9种可能的矩阵求导定义。

<img src="https://i.loli.net/2021/10/09/RlA1rJNKSY4Qoyn.png" alt="image-20211009160549220" style="zoom: 67%;" />



这9种求导中，标量对标量求导在高数中就有。我们先讨论上图中**标量y对向量x或矩阵X求导**，**向量或矩阵对标量求导**，以及**向量对向量求导**这5种情况。另外三种**向量对矩阵的求导**，**矩阵对向量的求导**，以及**矩阵对矩阵的求导**我们在后面再讲。



#### 3. 矩阵向量求导布局

因为求导结果可能会行向量，也可能会列向量。为了解决矩阵向量求导结果不唯一，我们引入**求导布局**。最简单的求导布局为**分子布局**和**分母布局**。

对于分子布局来说，我们求导结果的维度以分子为主，比如上面的对标量求导的例子，结果的维度和分子的维度是一样的。也就是说**y**为m维的列向量，那么求导结果`∂ y / ∂x`也是m维列向量。如果向量**y** 是一个m维的行向量，那么求导结果也是m维的行向量。

对于分母布局来说，我们求导结果的维度以分母为主，比如上面对标量求导例子，如果向量y是一个m维的列向量，那么求导结果**∂y / ∂x**是一个m维行向量。如果如果向量y是一个m维的行向量，那么求导结果**∂y / ∂x**是一个m维的列向量。

**对于分子布局和分母布局来说，两者相差一个转置。**

再举一个例子，标量y对矩阵X求导，那么如果按分母布局，则求导结果的维度和矩阵X的维度**m×n**是一致的。如果是分子布局，则求导结果的维度为**n×m**。

​	这样，对于标量对向量或者矩阵求导，向量或者矩阵对标量求导这4种情况，对应的分子布局和分母布局的排列方式已经确定了。稍微麻烦点的是向量对向量的求导，本文只讨论列向量对列向量的求导，其他的行向量求导只是差一个转置而已。比如m维列向量y对n维列向量x求导。它的求导结果在分子布局和分母布局各是什么呢？对于这2个向量求导，那么一共有mn个标量对标量的求导。求导的结果一般是排列为一个矩阵。如果是分子布局，则矩阵的**第一个维度以分子**为准，即结果是一个m×n的矩阵，如下：

![image-20211009163349291](https://i.loli.net/2021/10/09/qU7fx5VcnMpK8Qb.png)

上边这个按**分子布局的向量对向量求导的结果矩阵**，我们一般叫做**雅克比 (Jacobian)矩阵。**

如果是按分母布局，则求导的结果矩阵的**第一维度会以分母**为准，即结果是一个n×m的矩阵，如下：

![image.png](https://i.loli.net/2021/10/09/LGe7oiwfthZcNku.png)

上边这个**按分母布局的向量对向量求导的结果矩阵**，我们一般叫做**梯度矩阵**。

但是在机器学习算法原理的资料推导里，我们并没有看到说正在使用什么布局，也就是说布局被隐含了，这就需要自己去推演，比较麻烦。但是一般来说我们会使用一种叫混合布局的思路，即如果是**向量或者矩阵对标量求导**，则使用**分子布局**为准，如果是**标量对向量或者矩阵求导**，则以**分母布局**为准。对于**向量对对向量求导**，有些分歧，我的所有文章中会以**分子布局的雅克比矩阵为主**。

具体总结如下：

![image-20211009164442784.png](https://i.loli.net/2021/10/09/qAIpKmgbBhxfVNv.png)





#### 4. 标量对向量的求导

##### 1） 定义法求解标量对向量求导

标量对向量的求导严格来说是实值函数对向量的求导。对于一个给定的实值函数，如何求解∂y / ∂**x**呢？

首先我们想到的是基于矩阵求导的定义来做，由于所谓**标量对向量的求导**，其实就是**标量对向量里的每个分量分别求导**，最后把**求导的结果排列在一起，按一个向量表示**而已。

那么我们可以将实值函数对向量的每一个分量来求导，最后找到规律，得到求导的结果向量。

![image-20211009172258785.png](https://i.loli.net/2021/10/09/B3NcY1bHCqw9mlJ.png)

首先我们来看一个简单的例子：y=a<sup>T</sup>x，求解∂a<sup>T</sup>x / ∂x。

根据定义，我们先对**x**的第i个分量进行求导，这是一个**标量对标量的求导**，如下：

![image-20211009171216653.png](https://i.loli.net/2021/10/09/ZQ8L57oC1RcWEqD.png)

可见，对向量的第i个分量的求导结果就等于向量**a**的第i个分量。最后所有求导结果的分量组成的是一个n维向量，其实就是向量**a**，也就是

![image-20211009222034974](https://gitee.com/fanhang64/my_images/raw/master/2021/image-20211009222034974.png)

同理可得到

![image-20211009222107160](https://gitee.com/fanhang64/my_images/raw/master/2021/image-20211009222107160.png)

给一个测试，看能不能推导出：

![image-20211009222340775](https://gitee.com/fanhang64/my_images/raw/master/2021/image-20211009222340775.png)

其中，x<sup>T</sup>x 为`x1^2 + x2^2 + x3^2 + x4^2 + ... + xn^2`，第i项的求偏导为2x<sub>i</sub>，从而n个元素组成向量为**2x**。

##### 2） 标量对向量求导的一些基本法则

1) 常数对向量的求导结果为0

2) 线性法则：若f，g都是实值函数，c1和c2为常数，则

   ![image-20211009223830575](https://gitee.com/fanhang64/my_images/raw/master/2021/image-20211009223830575.png)

3) 乘法法则：如果f和g都是实战函数，则

   ![image-20211009223939881](https://gitee.com/fanhang64/my_images/raw/master/2021/image-20211009223939881.png)

   如果不是实值函数，则不能使用乘法法则。

4) 除法法则：如果f，g都是实值函数，且g(x) ≠ 0，则：

   ![image-20211009224120360](https://gitee.com/fanhang64/my_images/raw/master/2021/image-20211009224120360.png)



##### 3）用定义法求解标量对矩阵求导

标量对矩阵求导和标量对向量求导类似，只是结果是一个和自变量同型的矩阵。

例如：y=**a<sup>T</sup>Xb**，求解**∂a<sup>T</sup>Xb / ∂X**，其中，**a**是m维向量，**b**是n维向量，**X**是mxn的矩阵。

对矩阵**X**任意一个位置的X<sub>i</sub><sub>j</sub>求导，如下：

![image-20211010105651114](C:\Users\fanzone\AppData\Roaming\Typora\typora-user-images\image-20211010105651114.png)

　　即求导结果在(i,j)位置的求导结果是a向量第i个分量和b第j个分量的乘积，将所有的位置的求导结果排列成一个m×n的矩阵，即为ab<sup>T</sup>,这样最后的求导结果为：

![image-20211010105847348](C:\Users\fanzone\AppData\Roaming\Typora\typora-user-images\image-20211010105847348.png)


























































