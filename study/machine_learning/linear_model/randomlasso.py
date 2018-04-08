'''
利用Lasso和Adaptive Lasso建立模型
'''
from scipy.sparse import coo_matrix
from sklearn.utils import resample
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
import pandas as pd 
import random
import numpy as np 

class RandomLasso(object):
    def __init__(self,alpha = 1e-5,iterrations = 10,variable_size = 0.5,n_models = 100):
        self.alpha = alpha
        self.iterrations = iterrations
        self.n_models = n_models
        self.variable_size = variable_size        

    def bootstrap(self,x,y):
        #自助采样
        x_sparse = coo_matrix(x)
        x,x_sparse,y = resample(x,x_sparse,y)
        return x,y

    def choose_variable(self,x):
        new_var = random.sample(range(x.shape[1]),int(x.shape[1]*self.variable_size))
        return x[:,new_var],new_var

    def sample(self,x,y):
        sample_list = []
        var_list = []
        for i in range(self.n_models):
            x_ = x.copy()
            y_ = y.copy()
            x_,y_ = self.bootstrap(x_,y_)
            x_,new_var = self.choose_variable(x_)
            data_dict = {}
            data_dict['x'] = x_
            data_dict['y'] = y_
            sample_list.append(data_dict)
            var_list.append(new_var)
        return sample_list,var_list

    def first_fit(self,x,y):
        '''
        第一次拟合，旨在获取特征权重
        '''
        feature_importance = pd.DataFrame()
        feature_importance['index'] = [col for col in range(x.shape[1])]
        sample_list,var_list = self.sample(x,y)
        for i in range(len(sample_list)):
            sample = sample_list[i]
            variables = var_list[i]
            clf = Lasso()
            clf.fit(sample['x'],sample['y'])
            new_feature_importances = pd.DataFrame({'index':variables,'weight%d' % i:clf.coef_})
            feature_importance = pd.merge(feature_importance,new_feature_importances,how='left')        
        feature_importance = feature_importance.fillna(0)
        feature_importance['feature_importance'] = np.mean(feature_importance.drop('index',axis=1),axis=1)
        return feature_importance#['feature_importance'].values


if __name__ == "__main__":
    clf = RandomLasso(n_models=100)
    x,y = make_regression(n_samples=3,n_features = 4)
    print(x,y)
    x = clf.first_fit(x,y)
    print(x)