'''以波士顿房价为例'''
import tensorflow as tf
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

'''导入波士顿房价数据'''
boston = load_boston()

x_train,x_test,y_train,y_test = train_test_split(boston.data,boston.target,test_size = 0.2,random_state = 1)

#将训练数据规范化，两个维度都规范化
x_train = scale(x_train)
x_test = scale(x_test)
y_train = scale(y_train.reshape((-1,1)))
y_test = scale(y_test.reshape((-1,1)))


#定义一个添加网络层数的函数
def add_layer(inputs,input_size,output_size,activation_function=None):
    #将Weights定义为一个in_size行、out_size列的随机变量矩阵
    with tf.variable_scope("Weights"):
        Weights = tf.Variable(tf.random_normal(shape=[input_size,output_size]),name="weights")
    #因为在机器学习中，biases推荐值不为0，所以这里在0矩阵的基础上加了0.1
    with tf.variable_scope("biases"):
        biases = tf.Variable(tf.zero(shape=[1,output_size])+0.1,name="biases")
    #
    with tf.name_scope("Wx_plus_b"):
        Wx_plus_b = tf.matmul(inputs,Weights) + biases
    with tf.name_scope("dropout"):
        Wx_plus_b = tf.nn.dropout(Wx_plus_b,keep_prob = keep_prob_s)

    if activation_function is None:
        return Wx_plus_b
    else:
        with tf.name_scope("activation_function"):
            return activation_function(Wx_plus_b)

#定义占位符和网络层数
xs = tf.placeholder(shape=[None,x_train.shape[1]],dtype=tf.float32,name="inputs")
ys = tf.placeholder(shape=[None,1],dtype=tf.float32,name="y_true")
keep_prob_s = tf.placeholder(dtype=tf.float32)

with tf.name_scope("layer_1"):
    l1 = add_layer(xs,13,10,activation_function=tf.nn.relu)
with tf.name_scope("y_pred")
    pred = add_layer(l1,10,1)

pred = tf.add(pred,0,name="pred")



