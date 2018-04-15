# hmmlearn 简介

hmmlearn 原为sklearn中的模块，现在已经独立出来，需要另外安装。
```python
pip install hmmlearn
```
这个库在ubuntu环境下安装很顺利，但是win7下安装老是出错，如果遇到出错的情况，可以去[Python非官方第三方库网站](https://www.lfd.uci.edu/~gohlke/pythonlibs/)下载whl文件自行使用pip安装。


因为hmmlearn的官方文档写得不清不楚，所以要学习hmmlearn之前要对隐马尔可夫模型有一定的了解。

首先要知道隐马尔可夫模型的五大要素、三大假设、三大问题。

## 五大要素

S：状态值序列
O: 观察值序列
π：初始化概率
A: 状态转移矩阵
B: 给定状态下，观察值概率矩阵

而HMM模型本身具有三大参数$\lambda = (A,B,\pi)$

## 三大假设

1. 有限历史性假设：
    当前状态是否发生只与上一状态相关，即$p(s_i\left|s_{i-1}...s_1) = p(s_i\right|s_{i-1})$

2. 齐次性假设
    状态变化与具体时间无关，即$p(s_{i+1}|s_i) = p(s_{j+1}|s_j)$

3. 输出独立性假设
    输出值仅与当前状态有关

## 三大问题

1. 评估问题
    已知模型参数$\lambda = (A,B,\pi)$，计算某个观测序列O出现的概率。

2. 解码问题
    已知模型和观测序列，寻找与观测序列对应的可能性最大的状态序列。

3. 学习问题
    调整模型参数$\lambda = (A,B,\pi)$，使观测序列O的概率$P(O|\lambda)$最大。


在hmmlearn的官方文档中给出了使用hmmlearn分析股票隐藏状态的例子，相当于上述三大问题中的学习+解码问题。

> 因为例子中的雅虎金融数据获取不到，所以数据获取源使用了国内的Tushare，可以通过pip安装:
```python
pip install tushare
```

## 分析步骤

### 1. 导入数据

```python
import numpy as np
import matplotlib.pyplot as plt
import tushare as ts
from hmmlearn.hmm import GaussianHMM
data = ts.get_hist_data('600848',start = '2010-01-01',end='2017-12-31')
close_v = data['close'].values#当日收盘价格
volume = data['volume'].values#当日交易量
dates = np.array([i for i in range(data.shape[0])])#懒得处理日期了，这里直接使用阿拉伯数字代替日期
plt.plot(close_v,color = 'blue')
plt.show()
plt.savefig("H:/learning_notes/study/machine_learning/HMM/stocks.jpg")
```
股票数据如下图所示：

### 2. 处理数据

接下来需要将收盘价格转换成涨跌幅数据。
```python
diff = np.diff(close_v)#要训练的是收盘价格的变化值
dates = dates[1:]
close_v = close_v[1:]
volume = volume[1:]
X = np.column_stack([diff,volume])
diff = diff.reshape(-1,1)#一维数据需要进行处理
```

### 3. 建立模型

hmmlearn的API继承了sklearn一贯的简洁风格，初始化模型时只需要提供几个简单的参数就可以了，下面的n_components是状态序列中的状态种类数量，n_iter是迭代次数：
```python
model = GaussianHMM(n_components=2,n_iter=1000)
model.fit(diff)#训练模型————学习问题
hidden_states = model.predict(diff)#估计状态序列————解码问题
```

### 4. 绘制股票的不同状态

```python
for j in range(len(close_v)-1):
    for i in range(model.n_components):
        if hidden_states[j] == i:
            plt.plot([dates[j],dates[j+1]],[close_v[j],close_v[j+1]],color = colors[i])

plt.show()
plt.savefig("H:/learning_notes/study/machine_learning/HMM/hidden_states.jpg")
```
不同时刻的状态如下图所示，明显能看出该股票被分成了震荡与剧烈涨跌两种状态：



完整代码：
