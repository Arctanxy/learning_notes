import numpy as np 
import pandas as pd 
import collections
from sklearn.datasets import load_boston

class k_neighbor(object):
	def __init__(self,k = 3):
		self.k = k 
	#每个样本需要有一个标签或目标值
	#所以这个函数应该有regression和classifier两种模式
	def predict(self,train_data,test_data):
		distances = collections.defaultdict(list)#使用tuple存储
		for test in test_data:
			for train in train_data:
				distances[self.distance(test,train)] = train#因为list不能作为字典的key
		sorted_dict = sorted(distances.items(),key = lambda x:x[0],reverse = True)
		selected_data = [n[1] for n in sorted_dict[:self.k]]
		target_value = np.mean(selected_data,axis=0)
		print(target_value)
		
		
	def distance(self,p1,p2):
		return np.sqrt(np.sum([np.square(p1[i]-p2[i]) for i in range(len(p1))]))

#加载sklearn中自带的boston房价数据集
def read_boston_data():
	boston = load_boston()
	x = np.array(boston.data)
	y = np.array(boston.target)
	y = y.reshape(y.shape[0],1)
	return x,y
	
if __name__ == "__main__":
	x,y = read_boston_data()
	
	clf = k_neighbor(k = 1)
	clf.predict(train_data,x)