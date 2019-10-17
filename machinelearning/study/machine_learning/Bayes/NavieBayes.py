'''
朴素贝叶斯模型
'''
import pandas as pd 
import numpy as np 
from sklearn.datasets import load_iris
from collections import defaultdict
from sklearn.cross_validation import train_test_split

def load_data():
    '''
    加载鸢尾花数据
    '''
    data = load_iris()
    return data['data'],data['target']

class NBClassifier(object):
    def __init__(self):
        self.y = []#标签集合
        self.x = []#每个属性的数值集合
        self.py = defaultdict(float)#标签的概率分布
        self.pxy = defaultdict(dict)#每个标签下的每个属性的概率分布
        self.n = 5#分级的级数

    def prob(self,element,arr):
        '''
        计算元素在列表中出现的频率
        '''
        prob = 0.0
        for a in arr:
            if element == a:
                prob += 1/len(arr)
        if prob == 0.0:
            prob = 0.001
        return prob

    def get_set(self,x,y):
        self.y = list(set(y))
        for i in range(x.shape[1]):
            self.x.append(list(set(x[:,i])))#记录下每一列的数值集
        
    def fit(self,x,y):
        '''
        训练模型
        '''
        x = self.preprocess(x)
        self.get_set(x,y)
        #1. 获取p(y)
        for yi in self.y:
            self.py[yi] = self.prob(yi,y)
        #2. 获取p(x|y)
        for yi in self.y:
            for i in range(x.shape[1]):
                sample = x[y==yi,i]#标签yi下的样本
                #获取该列的概率分布
                pxy = [self.prob(xi,sample) for xi in self.x[i]]
                self.pxy[yi][i] = pxy
        print("train score",self.score(x,y))

    def predict_one(self,x):
        '''
        预测单个样本
        '''
        max_prob = 0.0
        max_yi = self.y[0]
        for yi in self.y:
            prob_y = self.py[yi]
            for i in range(len(x)):
                prob_x_y = self.pxy[yi][i][self.x[i].index(x[i])]#p(xi|y)
                prob_y *= prob_x_y#计算p(x1|y)p(x2|y)...p(xn|y)p(y)
            if prob_y > max_prob:
                max_prob = prob_y
                max_yi = yi
        return max_yi

    def predict(self,samples):
        '''
        预测函数
        '''
        samples = self.preprocess(samples)
        y_list = []
        for m in range(samples.shape[0]):
            yi = self.predict_one(samples[m,:])
            y_list.append(yi)
        return np.array(y_list)

    def preprocess(self,x):
        '''
        因为不同特征的数值集大小相差巨大，造成部分概率矩阵变得稀疏，需要进行数据分割
        '''
        for i in range(x.shape[1]):
            x[:,i] = self.step(x[:,i],self.n)
        return x
    
    def step(self,arr,n):
        '''
        分为n阶
        '''
        ma = max(arr)
        mi = min(arr)
        for i in range(len(arr)):
            for j in range(n):
                a = mi + (ma-mi)*(j/n)
                b = mi + (ma-mi)*((j+1)/n)
                if arr[i] >= a and arr[i] <= b:
                    arr[i] = j+1
                    break
        return arr
            
    def score(self,x,y):
        y_test = self.predict(x)
        score = 0.0
        for i in range(len(y)):
            if y_test[i] == y[i]:
                score += 1/len(y)
        return score

if __name__ == "__main__":
    x,y = load_data()
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.5,random_state = 100)
    clf = NBClassifier()

    clf.fit(x_train,y_train)
    
    score = clf.score(x_test,y_test)
    print('test score',score)