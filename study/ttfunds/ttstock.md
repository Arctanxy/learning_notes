# 使用LSTM预测股票趋势

参考上一篇利用LSTM预测美元汇率的文章，可以自行编写一个程序用于预测股票涨跌。

## 1. 工具

### 数据来源：Tushare

Tushare是一个开源财经数据包，其数据主要来自新浪财经和腾讯财经，调用起来非常方便。

### 模型工具：TensorFlow

TensorFlow不用过多介绍，本文中采用了自己搭建LSTM神经元的方式构建神经网络，以便与上一篇文章相呼应。

## 2. 导入数据

```Python
def get_data():
    '''
    获取历史收盘价格并归一化
    '''
    #加载数据
    stocks = ts.get_hist_data(code='600848',start='2010-01-01',end='2017-12-31')
    close_data = stocks['close'].values
    #归一化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(close_data.reshape(-1,1))
    return scaled_data
```

## 3. 处理数据

因为是做预测实验，所以需要将数据按时间分割，下面的代码中将数据分割成了每windowsize个x对应一个y。
```Python
def window_data(data,window_size):
    '''
    切割数据，长度比x:y = 7:1
    '''
    x = []
    y = []
    i = 0 
    while (i + window_size) <= data.shape[0] -1:
        x.append(data[i:i+window_size])
        y.append(data[i+window_size])
        i += 1
    assert len(x) == len(y)
    return x,y
```

分割完了之后要进行数据划分，划分成训练数据和测试数据，以便验证模型效果。

```Python
def train_test(x,y,test_size):
    '''
    划分训练与测试数据
    '''
    x_train = np.array(x[:int(len(x)*test_size)])
    y_train = np.array(y[:int(len(x)*test_size)])
    x_test = np.array(x[int(len(x)*test_size):])
    y_test = np.array(y[int(len(x)*test_size):])
    return x_train,y_train,x_test,y_test
```

## 4. 定义LSTM神经元

下面各个参数的具体功能可以参考[上一篇文章](https://arctanxy.github.io/2018/04/15/LSTM%E9%A2%84%E6%B5%8B%E6%B1%87%E7%8E%87%E5%8F%98%E5%8C%96/)。

```Python
def LSTM_cell(input,output,state):
    '''
    定义LSTM神经元
    '''
    #输入门
    weights_input_gate = tf.Variable(tf.truncated_normal([1,HIDDEN_LAYER],stddev=0.05))#tf.truncated_normal用于生成一定维度的正态分布数据
    weights_input_hidden = tf.Variable(tf.truncated_normal([HIDDEN_LAYER,HIDDEN_LAYER],stddev=0.05))
    bias_input = tf.Variable(tf.zeros([HIDDEN_LAYER]))

    #遗忘门
    weights_forget_gate = tf.Variable(tf.truncated_normal([1, HIDDEN_LAYER], stddev=0.05))
    weights_forget_hidden = tf.Variable(tf.truncated_normal([HIDDEN_LAYER, HIDDEN_LAYER], stddev=0.05))
    bias_forget = tf.Variable(tf.zeros([HIDDEN_LAYER]))

    #输出门
    weights_output_gate = tf.Variable(tf.truncated_normal([1, HIDDEN_LAYER], stddev=0.05))
    weights_output_hidden = tf.Variable(tf.truncated_normal([HIDDEN_LAYER, HIDDEN_LAYER], stddev=0.05))
    bias_output = tf.Variable(tf.zeros([HIDDEN_LAYER]))

    #记忆单元
    weights_memory_cell = tf.Variable(tf.truncated_normal([1, HIDDEN_LAYER], stddev=0.05))
    weights_memory_cell_hidden = tf.Variable(tf.truncated_normal([HIDDEN_LAYER, HIDDEN_LAYER], stddev=0.05))
    bias_memory_cell = tf.Variable(tf.zeros([HIDDEN_LAYER]))

    #定义各个门与状态
    input_gate = tf.sigmoid(tf.matmul(input, weights_input_gate) + tf.matmul(output, weights_input_hidden) + bias_input)
    forget_gate = tf.sigmoid(tf.matmul(input, weights_forget_gate) + tf.matmul(output, weights_forget_hidden) + bias_forget)
    output_gate = tf.sigmoid(tf.matmul(input, weights_output_gate) + tf.matmul(output, weights_output_hidden) + bias_output)
    memory_cell = tf.tanh(tf.matmul(input, weights_memory_cell) + tf.matmul(output, weights_memory_cell_hidden) + bias_memory_cell)
    state = state * forget_gate + input_gate * memory_cell
    output = output_gate * tf.tanh(state)
    return state, output
```

## 5. 训练模型

### (1) 定义参数

定义一些必须的模型参数


```Python
BATCH_SIZE = 7#模型每批次训练所接受的数据量
WINDOW_SIZE = 7#滑窗法切割
HIDDEN_LAYER = 256#隐藏层层数
CLIP_MARGIN = 4#用于控制梯度范围的参数
LEARNING_RATE = 0.001#步长
EPOCHS = 100#迭代次数
```

### (2) 定义变量

```Python
if __name__ == "__main__":
    data = get_data()
    x,y = window_data(data,WINDOW_SIZE)
    x_train,y_train,x_test,y_test = train_test(x,y,test_size=0.25)
    inputs = tf.placeholder(tf.float32,[BATCH_SIZE,WINDOW_SIZE,1])
    targets = tf.placeholder(tf.float32,[BATCH_SIZE,1])
    outputs = []
    #输出参数
    weights_output = tf.Variable(tf.truncated_normal([HIDDEN_LAYER,1],stddev=0.05))
    bias_output_layer = tf.Variable(tf.zeros([1]))
    for i in range(BATCH_SIZE):
        batch_state = np.zeros([1,HIDDEN_LAYER],dtype=np.float32)
        batch_output = np.zeros([1,HIDDEN_LAYER],dtype=np.float32)
        for j in range(WINDOW_SIZE):
            batch_state,batch_output = LSTM_cell(tf.reshape(inputs[i][j],(-1,1)),batch_state,batch_output)
        outputs.append(tf.matmul(batch_output,weights_output) + bias_output_layer)
    #定义模型损失
    losses = []
    for i in range(len(outputs)):
        losses.append(tf.losses.mean_squared_error(tf.reshape(targets[i],(-1,1)),outputs[i]))
    loss = tf.reduce_mean(losses)
    #定义优化器
    gradients = tf.gradients(loss,tf.trainable_variables())#计算梯度
    clipped,_ = tf.clip_by_global_norm(gradients,CLIP_MARGIN)#让梯度控制在一定范围内，防止梯度消失或者梯度爆炸
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    trained_optimizer = optimizer.apply_gradients(zip(gradients,tf.trainable_variables()))
```

### (3) 训练模型

```Python
    #训练模型
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(EPOCHS):
        trained_scores = []
        j = 0
        epoch_loss = []
        while (j+BATCH_SIZE) <= len(x_train):
            x_batch = x_train[j:j+BATCH_SIZE]
            y_batch = y_train[j:j+BATCH_SIZE]
            #c为cost，o为output
            o,c,_ = sess.run([outputs,loss,trained_optimizer],feed_dict={inputs:x_batch,targets:y_batch})
            epoch_loss.append(c)
            trained_scores.append(o)
            j += BATCH_SIZE
        
        if (i%30) == 0:
            print("Loss:{}".format(np.mean(epoch_loss)))
```

## 6. 检验模型

```Python
    #测试
    tests = []
    i = 0
    while i+BATCH_SIZE <= len(x_test):
        o = sess.run([outputs],feed_dict={inputs:x_test[i:i+BATCH_SIZE]})
        i += BATCH_SIZE
        tests.append(o)
    #因为得到的预测数据是一格一格的滑窗数据，有很多重复数据，需要进行处理
    tests_new = []
    for i in range(len(tests)):
        for j in range(len(tests[i][0])):
            tests_new.append(tests[i][0][j])
    #将结果一维化
    tests_new = np.squeeze(tests_new)
    print(len(data),len(tests_new))
    print(tests_new)
    fig = plt.figure()
    plt.plot(range(len(data)),data,color = 'r')
    plt.plot(range(len(data)-len(tests_new),len(data)),tests_new,color = 'g')
    plt.show()
    fig.savefig("H:/learning_notes/study/ttfunds/prediction.jpg")
```

最终预测结果为：

![](https://github.com/Arctanxy/learning_notes/blob/master/study/ttfunds/prediction.jpg?raw=true)