'''
使用tensorflow搭建线性回归模型
'''

import matplotlib.pyplot as plt 
import tensorflow as tf 
import numpy as np 

from numpy import genfromtxt
from sklearn.datasets import load_boston


def read_boston_data():
    boston = load_boston()
    x = np.array(boston.data)
    y = np.array(boston.target)

    return x,y

def feature_normalize(dataset):
    mu = np.mean(dataset,axis = 0)
    sigma = np.std(dataset,axis=0)

    return (dataset-mu)/sigma

def append_bias_reshape(x,y):#添加一个常数项，用于训练偏差
    n_training_samples = x.shape[0]#训练样本数量
    n_dim = x.shape[1]#特征数量
    f = np.reshape(np.c_[np.ones(n_training_samples),x],[n_training_samples,n_dim+1])
    l = np.reshape(y,[n_training_samples,1])
    
    return f,l

if __name__ == '__main__':
    
    #处理数据
    
    x,y = read_boston_data()
    norm_features = feature_normalize(x)
    f,l = append_bias_reshape(norm_features,y)
    n_dim = f.shape[1]

    rnd_indices = np.random.rand(len(f)) < 0.80#生成一个随机的布尔数组

    x_train = f[rnd_indices]
    y_train = l[rnd_indices]
    x_test = f[~rnd_indices]
    y_test = l[~rnd_indices]


    #tensorflow模型

    learning_rate = 0.01
    training_epochs = 6000
    cost_history = []
    test_history = []

    X = tf.placeholder(tf.float32,[None,n_dim])
    Y = tf.placeholder(tf.float32,[None,1])
    W = tf.Variable(tf.ones([n_dim,1]))

    init = tf.initialize_all_variables()

    y_ = tf.matmul(X,W)
    cost = tf.reduce_mean(tf.abs(y_-Y))
    training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    sess = tf.Session()
    sess.run(init)

    for epoch in range(training_epochs):
        sess.run(training_step,feed_dict={X:x_train,Y:y_train})
        c = sess.run(cost,feed_dict={X:x_train,Y:y_train})
        print(c)
        t = sess.run(cost,feed_dict={X:x_test,Y:y_test})
        print(t)
        cost_history.append(c)
        test_history.append(t)

    plt.plot(range(len(test_history)),test_history,color = 'green')
    plt.plot(range(len(cost_history)),cost_history,color = 'red')
    
    plt.axis([0,training_epochs,0,np.max(cost_history)])
    plt.show()
