import pandas as pd 
import numpy as np 
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt 

class linear_model(object):
	def __init__(self,max_time=100,intercept = False,normalize = True):
		self.alpha = 1e-5
		self.max_time = max_time
		self.theta = np.array([])
		self.mse = 0
		self.mae = 0
		self.intercept = intercept
	def fit(self,x,y):
		self.check_xy(x,y)
		if self.intercept == True:
			b = np.ones((x.shape[0],1))
			x = np.c_[x,b]
		else:
			pass
			
		if self.normalize == True:
			x,y = self.normalize(x),self.normalize(y)
		else:
			pass
			
		#self.theta = np.ones(x.shape[1])
		self.theta = np.array(np.dot(np.mat(np.dot(x.transpose(),x)).I,np.dot(x.transpose(),y)))
		#self.theta = self.theta.reshape(self.theta.shape[0],1)
		for i in range(self.max_time):
			y_pred = np.dot(x,self.theta)
			loss = y_pred-y 
			gradient = np.dot(x.transpose(),loss)/x.shape[0]
			self.theta = self.theta - self.alpha * gradient
			self.alpha = 1e-5 + 1/(i+1)
			print(np.mean(loss))
			#print(loss)
		self.mse = np.mean([l for l in loss*loss/(y*y)])
		self.mae = np.mean([l for l in np.abs(loss)/y])
		print(self.mae)
		print(self.theta)
	def pred(self,x):
		return np.dot(x,self.theta)
		
	def check_xy(self,x,y):
		if len(x.shape) < 2:
			raise Exception("X in wrong dimension")
		if len(y.shape) > 1 and y.shape[1] != 1:
			raise Exception("Y should be a 1D array")
		try:
			x = x.astype('float')
		except:
			print("X should not contain strings")
		if np.nan in y or np.inf in y:
			raise Exception("Y should not contain nan or inf")
			
		
	def center_data(self,dataset):
		'''
		中心化
		'''
		mu = np.mean(dataset,axis=0)
		return dataset-mu
	def normalize(self,dataset):
		'''
		正态化
		'''
		mu = np.mean(dataset,axis = 0)
		sigma = np.std(dataset,axis=0)
		return (dataset-mu)/sigma

#加载sklearn中自带的boston房价数据集
def read_boston_data():
	boston = load_boston()
	x = np.array(boston.data)
	y = np.array(boston.target)
	y = y.reshape(y.shape[0],1)
	return x,y
	
#将数据转化成（0，1）正态分布


if __name__ == "__main__":
	#x,y = read_boston_data()
	x = np.array([[1,1],[2,3],[3,4],[2,2]])
	y = np.array([[1],[2],[3],[4]])
	clf = linear_model(max_time = 10)
	clf.fit(x,y)
	print(clf.mse)
	print(clf.mae)