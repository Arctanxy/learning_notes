import pandas as pd 
import numpy as np 
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt 

FOLDER = 'H:/learning_notes/study/machine_learning/linear_model/'

class linear_regression(object):
	def __init__(self,max_time=1000,alpha = 1.0,intercept = False,normalize = True,batch_size = 10,method = 'matrix'):
		self.alpha = alpha
		self.max_time = max_time
		self.batch_size = batch_size
		self.theta = np.array([])
		self.mse = 0
		self.mae = 0
		self.intercept = intercept
		self.maes = []
		self.mses = []
		self.method = method
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
			
		self.theta = np.ones(x.shape[1])
		self.theta = self.theta.reshape(self.theta.shape[0],1)
		if self.method == 'sgd':
			self.GradientDescent(x,y)
		elif self.method == 'matrix':
			try:
				self.theta = np.dot(np.mat(np.dot(x.transpose(),x)).I,np.dot(x.transpose(),y))
				print('accuracy:',self.accuracy(x,y,self.theta,acc='mae'))
			except:
				print("请处理非独立特征")
	
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
			self.line_search(batch_data,batch_target,gradient)
			self.theta -= self.alpha * gradient
			#Show
			self.mse = self.accuracy(x,y,self.theta,acc='mse')
			self.mae = self.accuracy(x,y,self.theta,acc='mae')
			if self.mae < 2.0:
				self.mses.append(self.mse)
				self.maes.append(self.mae)
			print(self.mae)


	def line_search(self,batch_data,batch_target,gradient,threshold=1e-5,reduction_ratio=0.9):
		'''
		一维搜索
		其中threshold是一个经验参数，范围[0,0.5]
		reduction是缩小比,范围[0,1]
		'''
		while self.accuracy(batch_data,batch_target,self.theta - self.alpha * gradient) > self.accuracy(batch_data,batch_target,self.theta) - threshold * self.alpha * np.sum(gradient**2):
			self.alpha *= reduction_ratio


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

class Ridge(linear_regression):
	def __init__(self,lam = 0.2):
		linear_regression.__init__(self,max_time=1000,alpha = 1.0,intercept = False,normalize = True,batch_size = 10,method = 'matrix')
		self.lam = lam

	def fit(self,x,y):
		x,y = self.check_xy(x,y)
		if self.method == 'matrix':
			xtx = np.dot(x.transpose(),x)
			m = xtx.shape[0]
			Is = np.mat(np.eye(m))
			f1 = np.mat(xtx + self.lam * Is).I
			f2 = np.dot(x.transpose(),y)
			self.theta = np.dot(f1,f2)
			print(self.theta)
		else:
			print("to be continued……")
		print(self.accuracy(x,y,self.theta,acc='mae'))
	def predict(self,x):
		return np.array(np.dot(x,self.theta).transpose())[0]
		
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
	#clf = linear_regression(max_time = 1000,intercept=True,batch_size=1000)
	clf = Ridge(lam=0.2)
	clf.fit(x,y)
	'''clf.learning_curve(accuracy='mae')
	y_pred = clf.pred(x)
	plt.plot(range(len(y)),y,color = 'r')
	plt.plot(range(len(y)),y_pred,color = 'g')
	plt.savefig(FOLDER + 'predicted_data.png')
	plt.show()
	'''