# Random Lasso

Random Lasso是一种较为容易实现的集成线性回归算法，来自[Random Lasso](https://arxiv.org/pdf/1104.3398.pdf)

## 模型思路如下：

1. 从总样本中通过自助取样和随机采取特征的方式，得到N个样本（取样方式与随机森林的取样方式相同）；
2. 对着N个样本分别建立N个Lasso回归模型，根据模型中的系数，以计算均值的方式确定每个特征的权重；
3. 重新取样，步骤同1，但是特征取样的概率遵循特征权重；
4. 得到N个样本后，对每个建立一个Adaptive Lasso模型，以所有模型预测结果的均值为最终结果。


## 具体步骤

### 1. 模型简介

假设有样本：$(X_1,y_1),(X_2,y_2),...,(X_i,y_i),...,(X_n,y_n)$，其中$X_i = (x_{i1},...,x_{ip})^T$是一个p维向量，我们可以建立以下模型：

$$y_i = \beta_1 x_i1 + ... + \beta_p x_{ip} + \epsilon$$


其中的$\epsilon$是一个符合$N(0,1)$正态分布的常数，

Lasso回归与普通线性回归不同的是它在常规的误差函数中添加了一个L1范数作为惩罚项，以减小模型发生过拟合的可能性，同时因为L1范数的特性，Lasso也具备了特征筛选的能力，而这个特征筛选的能力既是优点，也是缺点。Lasso回归中需要优化的误差项如下：

$$\min_{\beta}\sum_{i=1}^{n}(y_i - \sum_{j=1}^{p}\beta_j x_{ij})^2 + \lambda \sum_{j=1}^{p}\left|\beta_j\right|$$

Lasso非常好用，至今还有不少学者在研究Lasso，因为Lasso还有不少的可优化空间，在实际使用中，Lasso主要有两大局限[Zou](https://mathscinet.ams.org/mathscinet-getitem?mr=2137327)：

1. 如果模型中有一些高度线性相关的特征，Lasso通常会从中选择一个或者一部分特征，而其他的线性相关特征都置为0。这样的特性在某些应用场景中可能会造成不好的影响。比如在基因序列分析中，基因的表达水平都会遵循一个共同的生物途径，所以往往会体现出高度相关性，但是这些基因又都能作用于生物过程，Lasso却只会从中选择一两个基因。理想的模型应该能选择所有的重要基因，去除无关紧要的基因。

2. 当p>n时，即特征数量大于样本数量时，Lasso最多只能选择n个特征。这样的特性在某些场景同样会出现问题，比如上面的基因分析问题，可能会导致最终的特征选择不足，很多有用的特征无法取到。


<文中还提到了弹性网络，但是因为与Random Lasso模型不太相关，我就略过了>


[Zou](https://mathscinet.ams.org/mathscinet-getitem?mr=2279469)对Lasso进行了修改，对每个特征使用不同的惩罚项，形成了新的自适应Lasso回归，其误差项如下：

$$\min_{\beta}\sum_{i=1}^{n}(y_i - \sum_{j=1}^{p}\beta_j x_{ij})^2 + \lambda \sum_{j=1}^{p} w_j \left|\beta_j\right|$$

其中的$w_j = \left| \hat{\beta}_j^ols \right|^{-r}$，$\hat{\beta}_j^ols$是$\beta_j$的最小二乘估计量，而r是一个正实数。

关于Adaptive Lasso的原理可以参看,[李锋，部分线性模型的AdaptiveLasso特征选择](http://www.oalib.com/paper/4762685#.Wslud_luYdU)。[基于迭代加权L1范数的稀疏阵列综合](https://wenku.baidu.com/view/436f16e8dd36a32d7275819f.html)


> 在Python 的sklearn库中，虽然没有adaptive lasso回归函数，但是却可以借助lasso实现adaptive lasso。具体实现将在下一篇文章中介绍。


### 2. RandomLasso

了解了以上知识后，下面介绍Random Lasso模型的具体步骤。

前面提到了Lasso能够筛选特征，但是这种筛选特征的方式有一个短板，就是对于一些有线性相关的重要特征，Lasso只会从中挑选一个或者一部分。

如果一些独立的数据都符合相同的分布，我们希望Lasso能够从不同的数据集中选取出这些重要因素的不同子集，通过随机选择特征集的方式，我们最终选择的重要特征集合就有可能会包含大部分甚至全部的有线性相关的重要特征。同时这种方式也解决了Lasso的第二个局限。






