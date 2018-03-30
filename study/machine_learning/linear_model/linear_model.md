# sklearn线性回归代码分析

## sklearn中的linear_regression()的思路

### 参数

1. fit_intercept:是否使用截距，默认为True；
2. normalize：是否进行正态化，默认为False，如果fit_intercept为False时会自动忽略此参数；
3. copy_x：是否复制x，默认为True，若设置为False则会对原x进行覆盖；
4. n_jobs：运算中将要使用到倒cpu核数

### linear_regression.fit()函数步骤

1. checkxy:
	检查x和y是否符合模型数据要求，如：
		y中是否有nan或者inf；
		x中如果有object类型，要想办法转换成数值，转换失败的话要抛出错误；
		x是否是二维矩阵型数据等
2. 查看sample_weight的初始值是否符合要求
	sample_weight是一维数组或者是一个标量
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
