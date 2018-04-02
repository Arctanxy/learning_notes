import pandas as pd 
import numpy as np 
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt 

FOLDER = 'H:/learning_notes/study/machine_learning/linear_model/'

class linear_regression(object):
	def __init__(self,max_time=1000,alpha = 1e-3,intercept = False,normalize = True,batch_size = 10):
		self.alpha = alpha
		self.max_time = max_time
		self.batch_size = batch_size
		self.theta = np.array([])
		self.mse = 0
		self.mae = 0
		self.intercept = intercept
		self.maes = []
		self.mses = []
	def fit(self,x,y):
		'''
		训练模型
		'''
		x,y = self.check_xy(x,y)
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
			
		self.theta = np.ones(x.shape[1])/100
		#self.theta = np.array(np.dot(np.mat(np.dot(x.transpose(),x)).I,np.dot(x.transpose(),y)))
		self.theta = self.theta.reshape(self.theta.shape[0],1)
		self.GradientDescent(x,y)
	
	def GradientDescent(self,x,y):
		'''
		梯度下降
		'''
		for i in range(self.max_time):
			#Train
			batch_index = np.random.randint(low=0,high=x.shape[0],size=self.batch_size)#随机取batch_size个样本
			batch_data = x[batch_index]
			batch_target = y[batch_index]
			y_pred = np.dot(batch_data,self.theta)
			loss = y_pred - batch_target 
			gradient = np.dot(batch_data.transpose(),loss)/batch_data.shape[0]
			#self.alpha = self.line_search(self.accuracy,x,y,self.alpha,gradient)
			
			new_theta = self.theta - self.alpha * gradient
			
			#一维搜索效果不好，以此代码代之
			if i <= 20:
				for j in range(10):
					if self.accuracy(batch_data,batch_target,self.theta,acc='mse') < self.accuracy(batch_data,batch_target,new_theta,acc='mse'):
						self.alpha *= 0.5
						new_theta = self.theta - self.theta * gradient


			self.theta = new_theta
			#print(self.alpha)

			#Show
			self.mse = self.accuracy(x,y,self.theta,acc='mse')
			self.mae = self.accuracy(x,y,self.theta,acc='mae')
			if self.mae < 2.0:
				self.mses.append(self.mse)
				self.maes.append(self.mae)
			print(self.mae)


	def line_search(self,f,x,y,alpha,gradient):
		'''
		一维搜索
		'''
		theta1 = self.theta
		theta2 = theta1 - alpha * gradient
		while True:
			if f(x,y,theta1) > f(x,y,theta2):
				theta3 = theta2 - alpha * gradient
				if f(x,y,theta2) < f(x,y,theta3):
					dis = (theta1 + theta3)/2
					best_alpha = np.mean((dis - self.theta)/(gradient+1))
					return best_alpha
				elif f(x,y,theta2) >= f(x,y,theta3):
                	#递推
					theta1 = theta2
					theta2 = theta3
					theta3 -= alpha*gradient
			elif f(x,y,theta1) < f(x,y,theta2):
				dis = (theta1 + theta2)/2
				alpha = alpha / 2
				theta2 = theta1 - alpha * gradient
				print(alpha)

	def accuracy(self,x,y,theta,acc = 'mse'):
		y_pred = np.dot(x,theta)
		loss = y_pred - y
		if acc == 'mse':
			return np.mean([l for l in loss*loss/(y * y)])
		elif acc == 'mae':
			return np.mean([l for l in np.abs(loss)/y])
		else:
			raise Exception("Wrong Accuracy!")
	

	def pred(self,x):
		'''
		预测数据
		'''
		if self.intercept == False:
			return np.dot(x,self.theta)
		else:
			b = np.ones((x.shape[0],1))
			x = np.c_[x,b]
			return np.dot(x,self.theta)

	def learning_curve(self,accuracy  = 'mae'):
		'''
		绘制学习曲线
		'''
		if accuracy == 'mae':
			plt.plot(range(len(self.maes)),self.maes,color = 'g')
			#plt.savefig(FOLDER + 'learning_curve.png')
			plt.show()
		elif accuracy == 'mse':
			plt.plot(range(len(self.mses)),self.mses,color = 'r')
			#plt.savefig(FOLDER + 'learning_curve.png')
			plt.show()
		else:
			raise Exception("Wrong accuracy")

	def check_xy(self,x,y):
		'''
		检查x与y的数据格式是否符合要求
		'''
		if type(y) == pd.core.series.Series:
			y = y.values
			x = x.values
		y = y.reshape(y.shape[0],1)
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
		return x,y
			
		
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
	return x,y
	


if __name__ == "__main__":
	x,y = read_boston_data()
	#x = np.array([[1,1],[2,3],[3,4],[2,2],[4,5]])
	#x = np.array([[1],[2],[4],[2],[5]])
	#y = np.array([[1],[2],[3],[4],[6]])
	clf = linear_regression(max_time = 1000,intercept=True,batch_size=1000)
	clf.fit(x,y)
	clf.learning_curve(accuracy='mae')
	y_pred = clf.pred(x)
	plt.plot(range(len(y)),y,color = 'r')
	plt.plot(range(len(y)),y_pred,color = 'g')
	plt.savefig(FOLDER + 'predicted_data.png')
	plt.show()
	