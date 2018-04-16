import pandas as pd 
import numpy as np 
from sklearn.datasets import load_iris
from collections import defaultdict

def load_data():
    data = load_iris()
    print(data)
    return data['data'],data['target']

class NBClassifier(object):
    def __init__(self):#一般不需要什么参数，sklearn中使用了先验概率为参数，也可以不用
        self.class_prior_ = None#每个类的概率
        self.class_count_ = None#每个类下的训练样本数量 
        self.theta_ = None#每个类下的每个特征均值
        self.sigma_ = None#每个类下的每个特征的方差
        pass
    
    def prob(self,arr):
        count = defaultdict(0)
        for a in arr:
            count[a] += 1/len(arr)
        return count

    def fit(self,x,y):
        #x和y都是numpy array类型
        #1. 计算每个类别的概率
        py = self.prob(y)
        #2. 计算每个特征在每个类下出现的概率
        pxs = []
        for yi in set(y):
            ids = y==yi
            sample = x[ids,:]
            for i in range(x.shape[1]):
                
            pxyi = self.prob(sample)
        
        pass
    def predict(self,x):
        '''
        p(yi|x) = p(x|yi)*p(yi)/p(x) = p(x1|yi)*p(x2|yi)*...*p(xn|yi)*p(yi)/p(x)
        因为对同一个样本来说p(x)是固定的，所以只需对比P = p(x1|yi)*p(x2|yi)*...*p(xn|yi)*p(yi)，得到使P最大的yi即为预测标签
        '''

if __name__ == "__main__":
    x,y = load_data()