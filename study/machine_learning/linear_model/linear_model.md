# 常见的线性回归算法实现手段

## 1. 矩阵运算

	在训练数据
	（1）各列数据线性独立
	（2）样本数量多于特征数量
	的前提下，可以使用矩阵形式计算线性回归的系数
	参考http://bourneli.github.io/linear-algebra/2016/03/03/linear-algebra-04-ATA-inverse.html

$$ Xw = Y $$
$$ X^TXw = X^TY$$
$$ w = (X^TX)^{-1}X^TY$$
	公式就这么简单。
## 2. 梯度下降

### 步长的确定

	在梯度下降算法中，最终结果可能有三种：（1）结果超出实数界（2）在某个范围内震荡（3）收敛
	
	在这个过程中起到决定性作用的参数就是步长alpha，从之前的例子中可以看到，固定步长的梯度下降函数对不同的数据表现出的结果差别很大，不同训练集中需要的步长可能会相差几个数量级。
	
	所以网络上常见的设置alpha = 0.001 + 1/(i+1)的这种给定初始值之后再逐步减小alpha的方法并没有实际应用价值，因为模型往往在alpha衰减到合适范围之前就已经超出实数界。

	**在训练过程中，必须根据实际数据动态调整步长**，调整步长可以使用line-search（线搜索、一维搜索）或者利用李普希兹常数。

#### （1） line-search

	line-search本身就是一种优化算法，在这里我们将实现的是效果较好的 gradient descent with backtracking line search。与之对应的还有gradient descent with exact line search,gradient descent with constant learning rate和 gradient descent using Newton's method。

	本文中使用的是backtracking line search。参考https://blog.csdn.net/u014791046/article/details/50831017
	http://www.hanlongfei.com/%E5%87%B8%E4%BC%98%E5%8C%96/2015/09/29/cmu-10725-gradient/

	**但是添加了line-search之后也只是能保证收敛而已，至于收敛到什么位置，还是比较随缘**

#### （2） 李普希兹连续

	对于在实数集的函数$f$，若存在常数K，使得 $|f(a)-f(b)| <= K|a-b|$ ，则称$f$符合李普希兹条件，对于$f$最小的常数$K$成为$f$的李普希兹常数，选择$1/K$作为学习步长（学习率）可以保证学习过程收敛。

很明显，用梯度下降训练线性回归模型不如矩阵算法简单易用。但是上面的矩阵算法的训练效果不如scikitlearn的linear_regression()效果好，scikitlearn是如何做到的呢？下面跟我一起来看一下scikitlearn中的linear_regression()源码。

## 3. sklearn线性回归代码分析

# sklearn有可能是可选的加权线性回归，代码还没看完，待确定

### sklearn中的linear_regression()的思路

#### 参数

1. fit_intercept:是否使用截距，默认为True；
2. normalize：是否进行正态化，默认为False，如果fit_intercept为False时会自动忽略此参数；
3. copy_x：是否复制x，默认为True，若设置为False则会对原x进行覆盖；
4. n_jobs：运算中将要使用到倒cpu核数

#### linear_regression.fit()函数步骤

1. checkxy:
	检查x和y是否符合模型数据要求，如：
		y中是否有nan或者inf；
		x中如果有object类型，要想办法转换成数值，转换失败的话要抛出错误；
		x是否是二维矩阵型数据等
2. 查看sample_weight的初始值是否符合要求
	sample_weight是一维数组或者是一个标量,sample_weights是样本权重，默认为None即每个样本带有均等的权重。
3. 数据预处理
	主要是进行中心化，正态化，
	如果是无截距训练的话，无需进行中心化，但是正态化还是必须的。
4. 缩放数据以应用样本权重
	将x，y分别乘以对角矩阵形式的sample_weight，使不同样本具有不同的权重
	```python
	from scipy import sparse
	from sklearn.utils.extmath import safe_sparse_dot
	import numpy as np 
	def _rescale_data(X, y, sample_weight):
		"""Rescale data so as to support sample_weight"""
		n_samples = X.shape[0]
		sample_weight = sample_weight * np.ones(n_samples)
		sample_weight = np.sqrt(sample_weight)
		sw_matrix = sparse.dia_matrix((sample_weight, 0),
                                  shape=(n_samples, n_samples))#将weight转化为对角方阵
		X = safe_sparse_dot(sw_matrix, X)
		y = safe_sparse_dot(sw_matrix, y)
		return X, y
	```
	
5. 判断x是否是稀疏矩阵
	如果是：
		判断y是否只有1列：
			如果是：
				用scipy中lsqr()函数求解稀疏矩阵方程
			如果不是：
				根据y的列数进行多线程稀疏矩阵方程求解
	如果不是：
		使用scipy中的lstsq()进行最小二乘法求解
		（很抱歉scipy的lstsq()函数的具体逻辑我没看懂……）


## 一维搜索（又名线搜索 line-search)

https://blog.csdn.net/qq_39422642/article/details/78826175

**线搜索的参数不是theta而是alpha**


ridge和lasso模型的实现可以参考http://python.jobbole.com/88799/
以及http://f.dataguru.cn/thread-598486-1-1.html


## 岭回归的矩阵运算推导

在进行推导之前，需要先了解有关矩阵的迹和矩阵微分的内容

### 矩阵的迹：
	$tr(A)$，表示$n*n$矩阵A的对角线元素之和

### 矩阵范数

矩阵的L2范数$||A||^2_2 = \lambda$，其中$\lambda$是矩阵$A^TA$的最大特征值。

因此，对于一维向量而言就有$||a||^2_2 = a^Ta$，因为$a^Ta$是一个标量，所以$a^Ta$的值就是其最大特征值。

### 矩阵微分


(矩阵微分公式参考 https://blog.csdn.net/u010976453/article/details/54381248)


目标：$F = min(||Y-XW||^2_2 + \lambda||W||^2_2)$

$$ F = (Y-XW)^T(Y-XW) + \lambda W^TW$$

$$ F = (Y^T - W^TX^T)(Y-XW) + \lambda W^TW$$

$$ F = Y^TY - Y^TXW - W^TX^TY + W^TX^TXW + \lambda W^TW$$

对F进行求导得到：



$$ \frac{\partial F}{\partial W} = - Y^TX - Y^TX + (X^TX + X^TX)W + \lambda W$$

$$ \frac{\partial F}{\partial W} = -2 Y^TX + 2X^TXW + \lambda W$$

令$\frac{\partial F}{\partial W} = 0$，有：

$$\frac{\partial F}{\partial W} = -2 Y^TX + 2X^TXW + \lambda W = 0$$

$$(X^TX + \frac{\lambda I}{2})W = Y^TX$$

$$W = (X^TX + \frac{\lambda I}{2})^{-1}Y^TX$$

推导地址： https://blog.csdn.net/computerme/article/details/50486937

