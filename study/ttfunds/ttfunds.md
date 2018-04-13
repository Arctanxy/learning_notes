# 使用循环神经网络预测汇率涨跌

本文中讲简单地介绍如何使用时间序列分析的方法预测汇率变化。

## 序列问题

首先介绍一下序列问题，常见的机器学习问题都是是一对一的模型，如下图所示：

[一对一模型](https://cdn-images-1.medium.com/max/1600/0*7AIMLPm1e7hgGolz.)

在这个例子中，我们将一个输入数据传入到模型中，然后模型会根据传入数据生成一个结果，像线性回归，分类问题甚至图像分类卷积神经网络都属于这种类型。

这种模式经过修改可以用于处理一对多模型，如下图所示，模型的输出数据会作为新的输入数据传入回神经网络中，从而产生一系列的值，这种神经网络叫做循环神经网络。

[循环神经网络](https://cdn-images-1.medium.com/max/1600/0*QFWZFOLMH4EyyZxu.)

对于序列型的输入数据，循环神经网络的工作方式如下图所示，每个循环网络神经元的输出都会进入下一个神经元，作为下一个神经元的一部分输入数据：

[循环神经网络处理序列问题](https://cdn-images-1.medium.com/max/1600/0*x1vmPLhmSow0kzvK.)

上面的网络中的每个神经元都使用同一个公式：

$$Y_t = tanh(wY_{t-1} + ux_t)$$

其中$Y_t$是当前神经元的输出，$Y_{t-1}$是上一个神经元的输出数据，$x_t$是当前神经元的原始输入，$w$和$u$都是权重参数。

可以通过简单地堆叠神经元来构建一个深层循环神经网络，但是简单的循环神经网络只能处理短时间记忆，对于需要长时间记忆的问题准确度会下降。

对于需要长时间依赖的序列分析问题，我们可以使用lstm神经网络来处理。

## LSTM神经网络简介

在上世纪九十年代，[Sepp Hochreiter和Jurgen Schmidhuber](http://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735)提出了LSTM神经网络，解决了传统的循环神经网络、隐马尔可夫模型和其他序列模型对长时间跨度不敏感的问题。

[LSTM神经元内部结构](https://cdn-images-1.medium.com/max/800/0*_rC7UKSazzfOkpFZ.)

LSTM神经元在传统循环神经网络中添加了一些逻辑门，它们的功能如下：
### 1. 遗忘门

$$f_t = \sigma(W_f[h_{t-1},x_t]+b_f)$$

遗忘门会接受来自上一神经元的输出$h_{t-1}$以及当前神经元的输入$x_t$，通过线性变换之后传入sigmod函数，得到一个介于0到1的数字，这个数字可以认为是门的开度。这个数字会与内部状态相乘，所以这个门成为遗忘门，因为如果$f_t$为0时，当前的内部状态会被完全丢弃，如果$f_t$为1时，当前状态会保持完好无损。

### 2. 输入门

$$i_t = \sigma(W_i[h_{t-1},x_t]+b_i$$

输入门中同样会取上一神经元的输出$h_{t-1}$以及当前神经元的输入$x_t$，经过线性变换后传入sigmoid函数，返回介于0到1到值，该值会与记忆单元的输出值相乘，记忆单元的方程如下：

$$C_t = tanh(W_c[h_{t-1},x_t]+b_c$$

这一层会对当前输入和上一层输入的线性结果进行双曲正切函数处理，其返回的向量将会被添加到内部状态中。

内部状态通过如下规则更新：

$$C_t = f_t * C_{t-1} + i_t * C_t$$

上一层的状态$C_{t-1}$会与遗忘门的返回值相乘，当前的$C_t$的$i_t$倍相加，得到该神经元的最终状态$C_t$，$C_t$会传输到输出门进行计算。

### 3. 输出门

$$O_t = \sigma(W_o[h_{t-1},x_t] + b_o)$$

$$h_t = O_t * tanh(C_t)$$

输出门决定了传入最终output的内部状态的比例，作用方式与前面两种门相似。

以上的三个们分别有独自的权重和偏置，也就是说神经网络将会学习保留多少往期数据，保留多少输入数据，传输多少内部状态。

在循环神经网络中，你不止向网络传入数据，还需要传入上一时刻的状态，这种特性对于需要联系上文的自然语言处理非常有效。

另外循环神经网络还可以用在时间序列分析、视频处理、语音识别等领域。

本文中将简单地介绍一下循环神经网络在汇率预测方面的应用。

将使用的数据是美元与印度卢比的在1980年1月2日到2017年8月10日这段时间内的汇率，共有13730条数据，图像如下：

[汇率变化](https://cdn-images-1.medium.com/max/800/0*UYHLdtUFPTM7YPs6.)

## 模型搭建

### 训练测试数据划分

以1980/1/2到2009/12/31期间为训练数据，2010/1/1到2017/8/10期间为测试数据。

[数据划分](https://cdn-images-1.medium.com/max/800/0*jXH_D2Zd8TOmXa1H.)

分组之前要对数据进行归一化。

### 尝试全连接神经网络


### 使用LSTM神经网络预测

#### 导入需要用到到库

```python
import tensorflow as tf
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
```
#### 步骤一、加载数据并进行归一化
```python
data = pd.read_csv('data.csv')
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data['Close'].values,reshape(-1,1))
```
#### 使用滑窗法切割数据集
```python
def window_data(data,window_size):
    x = []
    y = []
    i = 0
    while (i+window_size) <= len(data) -1:
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])#使用window_size个数据预测后面的一个数据
        i += 1
    assert len(X) == len(y)
    return X,y
X,y = window_data(scaled_data,7)

X_train  = np.array(X[:700])
y_train = np.array(y[:700])

X_test = np.array(X[700:])
y_test = np.array(y[700:])
```

#### 设置神经网络参数

```python
#模型参数
batch_size = 7#每次传入的样本数量
window_size = 7#滑窗法的窗口大小
hidden_layer = 256#神经元数量
clip_margin = 4#为防止梯度爆炸而采用的clipper的参数
learning_rate = 0.001#步长
epochs = 200#训练次数

#定义输入输出
inputs = tf.placeholder(tf.float32,[batch_size,window_size,1])
targets = tf.placeholder(tf.float32,[batch_size,1])

#定义权重和偏置
# LSTM weights
#Weights for the input gate
weights_input_gate = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05))
weights_input_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
bias_input = tf.Variable(tf.zeros([hidden_layer]))

#遗忘门权重
weights_forget_gate = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05))
weights_forget_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
bias_forget = tf.Variable(tf.zeros([hidden_layer]))

#输出门权重
weights_output_gate = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05))
weights_output_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
bias_output = tf.Variable(tf.zeros([hidden_layer]))

#记忆单元权重
weights_memory_cell = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05))
weights_memory_cell_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
bias_memory_cell = tf.Variable(tf.zeros([hidden_layer]))

#输出层权重
weights_output = tf.Variable(tf.truncated_normal([hidden_layer, 1], stddev=0.05))
bias_output_layer = tf.Variable(tf.zeros([1]))
```
#### 定义神经元

```python
def LSTM_cell(input, output, state):
    
    input_gate = tf.sigmoid(tf.matmul(input, weights_input_gate) + tf.matmul(output, weights_input_hidden) + bias_input)
    
    forget_gate = tf.sigmoid(tf.matmul(input, weights_forget_gate) + tf.matmul(output, weights_forget_hidden) + bias_forget)
    
    output_gate = tf.sigmoid(tf.matmul(input, weights_output_gate) + tf.matmul(output, weights_output_hidden) + bias_output)
    
    memory_cell = tf.tanh(tf.matmul(input, weights_memory_cell) + tf.matmul(output, weights_memory_cell_hidden) + bias_memory_cell)
    
    state = state * forget_gate + input_gate * memory_cell
    
    output = output_gate * tf.tanh(state)
    return state, output
```



原文地址：
https://blog.statsbot.co/time-series-prediction-using-recurrent-neural-networks-lstms-807fa6ca7f

