import pandas as pd 
import numpy as np 
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt 

class linear_model(object):
	def __init__(self,alpha=1e-5,max_time=100):
		self.alpha = alpha
		self.max_time = max_time
		self.theta = np.array([])
		self.mse = 0
		self.mae = 0
	def fit(self,x,y):
		self.theta = np.ones(x.shape[1])
		self.theta = self.theta.reshape(self.theta.shape[0],1)
		for i in range(self.max_time):
			y_pred = np.dot(x,self.theta)
			loss = y_pred-y 
			gradient = np.dot(x.transpose(),loss)/x.shape[0]
			self.theta = self.theta - self.alpha * gradient
			#self.alpha = self.alpha*0.99
		self.mse = np.mean([l for l in loss*loss/(y*y)])
		self.mae = np.mean([l for l in np.abs(loss)/y])
	def pred(self,x):
		return np.dot(x,self.theta)

#加载sklearn中自带的boston房价数据集
def read_boston_data():
	boston = load_boston()
	x = np.array(boston.data)
	y = np.array(boston.target)
	y = y.reshape(y.shape[0],1)
	return x,y
	
#将数据转化成（0，1）正态分布
def feature_normalize(dataset):
    mu = np.mean(dataset,axis = 0)
    sigma = np.std(dataset,axis=0)

    return (dataset-mu)/sigma

if __name__ == "__main__":
	x,y = read_boston_data()
	x = feature_normalize(x)
	clf = linear_model(max_time = 10000)
	clf.fit(x,y)
	print(clf.mse)
	print(clf.mae)