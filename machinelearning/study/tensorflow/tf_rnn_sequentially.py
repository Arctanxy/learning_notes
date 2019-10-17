import tensorflow as tf 
from tensorflow.contrib import rnn
import numpy as np 
import matplotlib.pyplot as plt 

def RNN(x,weights,biases,n_input,n_steps,n_hidden):
    '''输入数据格式(batch_size,n_steps,n_input)
    要求格式(n_steps,batch_size,n_input)
    '''
    x = tf.transpose(x,[1,0,2])
    x = tf.reshape(x,[-1,n_input])
    
    #将x分割成n_steps段数据
    x = tf.split(x,n_steps,axis=0)

    #定义lstm神经元
    lstm_cell = rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)
    
    #定义输出
    outputs,_ = rnn.static_rnn(lstm_cell,x,dtype=tf.float32)#此函数会返回outputs和states，states代表状态，这里用不到。

    return tf.nn.bias_add(tf.matmul(outputs[-1],weights['out']),biases['out'])

def generate_sample(f = None,t0 = None,batch_size = 1,samples = 100,predict = 50,):
    '''
    参数简介：
    f:时间序列的频率，None表示随机
    t0:时间序列的偏移量，None表示随机
    batch_size:要生成的时间序列数量
    predict:要生成的未来样本数量
    samples:要生成的当前样本数量
    return:
        包含过去的key,values和未来的key,values的元组，每一行代表一个时间序列的batch
    '''
    #生成训练数据
    T = np.empty((batch_size,samples))
    Y = np.empty((batch_size,samples))
    #生成测试数据
    FT = np.empty((batch_size,predict))
    FY = np.empty((batch_size,predict))
    for i in range(batch_size):
        t = np.arange(0,samples+predict)/100#自变量序列
        _t0 = t0
        if _t0 is None:
            t0 = np.random.rand() * 2* np.pi #偏移量，随机偏移n个周期
        else:
            t0 = _t0 + i/float(batch_size)#如果生成多组数据的话，将数据区间逐渐平移，以产生不同的数据
        
        freq = f
        if freq is None:
            freq = np.random.rand() * 3.5 + 0.5#随机生成一个频率

        y = np.sin(2*np.pi * freq * (t+t0))#计算y
        

        #分别截取训练数据和测试数据
        T[i,:] = t[0:samples]
        Y[i,:] = y[0:samples]

        FT[i,:] = t[samples:samples + predict]
        FY[i,:] = y[samples:samples + predict]


    return T,Y,FT,FY
        





if __name__ =="__main__":
    #参数
    learning_rate = 0.001
    training_iters = 30000
    batch_size = 50
    display_step = 100

    #网络参数
    n_input = 1
    n_steps = 100
    n_hidden = 150
    n_outputs = 50

    #定义计算图的input占位符
    x = tf.placeholder("float",[None,n_steps,n_input])
    y = tf.placeholder("float",[None,n_outputs])

    #权重和偏置
    weights = {
        'out':tf.Variable(tf.random_normal([n_hidden,n_outputs]))
    }
    biases = {
        'out':tf.Variable(tf.random_normal([n_outputs]))
    }

    pred = RNN(x,weights,biases,n_input,n_steps,n_hidden)

    #定义误差和优化器
    individual_losses = tf.reduce_sum(tf.squared_difference(pred,y))
    loss = tf.reduce_mean(individual_losses)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    #初始化
    init = tf.global_variables_initializer()

    #运行
    with tf.Session() as sess:

        sess.run(init)

        step = 1

        #训练
        while step * batch_size < training_iters:
            _,batch_x,_,batch_y = generate_sample(f=None,t0=None,batch_size=batch_size,samples=n_steps,predict=n_outputs)
            batch_x = batch_x.reshape((batch_size,n_steps,n_input))
            batch_y = batch_y.reshape((batch_size,n_outputs))

            sess.run(optimizer,feed_dict = {x:batch_x,y:batch_y})

            if step % display_step == 0:
                loss_value = sess.run(loss,feed_dict = {x:batch_x,y:batch_y})
                print("Iter" + str(step * batch_size) + ",Minibatch loss=" + "{:.6f}".format(loss_value))
            step += 1
        print("Finished")

        #测试
        n_tests = 3
        for i in range(1,n_tests+1):
            plt.subplot(n_tests,1,i)
            t,y,next_t,expected_y  = generate_sample(f=i,t0=None,samples=n_steps,predict=n_outputs)

            test_input = y.reshape((1,n_steps,n_input))
            prediction = sess.run(pred,feed_dict = {x:test_input})

            #去除batch_size这一维度
            #squeeze()用于从shape中删除一维条目
            t = t.squeeze()
            y = y.squeeze()
            next_t = next_t.squeeze()
            prediction = prediction.squeeze()

            plt.plot(t,y,color = 'black')
            plt.plot(np.append(t[-1],next_t),np.append(y[-1],expected_y),color = 'green',linestyle=':')
            plt.plot(np.append(t[-1],next_t),np.append(y[-1],prediction),color = 'red')

        plt.show()