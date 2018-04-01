import pandas as pd 
import numpy as np 
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt 

FOLDER = 'H:/learning_notes/study/machine_learning/linear_model/'

class linear_model(object):
	def __init__(self,max_time=100,alpha = 1e-5,intercept = False,normalize = True):
		self.alpha = alpha
		self.max_time = max_time
		self.theta = np.array([])
		self.mse = 0
		self.mae = 0
		self.intercept = intercept
		self.maes = []
		self.mses = []
	def fit(self,x,y):
		self.check_xy(x,y)
		if self.intercept == True:
			b = np.ones((x.shape[0],1))
			x = np.c_[x,b]
		else:
			pass
			
		if self.normalize == True:
			x = self.normalize(x)#一般只对x进行正态化，y如果是偏态分布的话，常见的是取对数处理
			#y = self.normalize(y)
		else:
			pass
			
		self.theta = np.ones(x.shape[1])
		#self.theta = np.array(np.dot(np.mat(np.dot(x.transpose(),x)).I,np.dot(x.transpose(),y)))
		self.theta = self.theta.reshape(self.theta.shape[0],1)
		for i in range(self.max_time):
			y_pred = np.dot(x,self.theta)
			loss = y_pred-y 
			gradient = np.dot(x.transpose(),loss)/x.shape[0]
			self.theta = self.theta - self.alpha * gradient
			#self.alpha = 1e-7 + 1/(i+1)
			if np.mean(gradient)* self.alpha > 3  * np.mean(self.theta):#如果出现震荡则缩小alpha
				self.alpha = self.alpha *0.8
			new_loss = np.dot(x,self.theta) - y

			self.mse = np.mean([l for l in loss*loss/(y*y)])
			self.mae = np.mean([l for l in np.abs(loss)/y])
			if self.mae < 2.0:
				self.mses.append(self.mse)
				self.maes.append(self.mae)
			print(self.mae)
		print(self.mae)
	
	def pred(self,x):
		if self.intercept == False:
			return np.dot(x,self.theta)
		else:
			b = np.ones((x.shape[0],1))
			x = np.c_[x,b]
			return np.dot(x,self.theta)

	def learning_curve(self,accuracy  = 'mae'):
		if accuracy == 'mae':
			plt.plot(range(len(self.maes)),self.maes,color = 'g')
			plt.savefig(FOLDER + 'learning_curve.png')
			plt.show()
		elif accuracy == 'mse':
			plt.plot(range(len(self.mses)),self.mses,color = 'r')
			plt.savefig(FOLDER + 'learning_curve.png')
			plt.show()
		else:
			raise Exception("Wrong accuracy")

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
	x,y = read_boston_data()
	#x = np.array([[1,1],[2,3],[3,4],[2,2],[4,5]])
	#x = np.array([[1],[2],[4],[2],[5]])
	#y = np.array([[1],[2],[3],[4],[6]])
	clf = linear_model(max_time = 10000,intercept=True)
	clf.fit(x,y)
	clf.learning_curve(accuracy='mae')
	y_pred = clf.pred(x)
	plt.plot(range(len(y)),y,color = 'r')
	plt.plot(range(len(y)),y_pred,color = 'g')
	plt.savefig(FOLDER + 'predicted_data.png')
	plt.show()
	