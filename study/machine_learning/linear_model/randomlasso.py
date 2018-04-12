'''
利用Lasso和Adaptive Lasso建立模型

目前效果一般，不如单纯的Lasso，但是比adaptive lasso效果要好一点
尚未实现功能：
    1. 使用袋外数据验证每个基础模型，然后使用得到的准确率作为权重，用于计算最后的系数的加权平均值
'''
from scipy.sparse import coo_matrix
from sklearn.utils import resample
from sklearn.datasets import make_regression,load_boston
from sklearn.linear_model import Lasso
import pandas as pd 
import random
import numpy as np 
from adaptive_lasso import AdaptiveLasso

class RandomLasso(object):
    def __init__(self,alpha = 1e-5,iterrations = 10,variable_size = 0.5,n_models = 100):
        self.alpha = alpha
        self.iterrations = iterrations
        self.n_models = n_models
        self.variable_size = variable_size        
        self.models = []
        self.variables = []
        self.coef_ = None
    def bootstrap(self,x,y):
        #自助采样
        x_sparse = coo_matrix(x)
        x,x_sparse,y = resample(x,x_sparse,y)
        return x,y

    def lahiri(self,x,feature_importance=None):
        if x.shape[1] != len(feature_importance):
            raise Exception("x与feature_importance长度不匹配")
        #Mmid = np.percentile(np.abs(feature_importance),self.variable_size*100)
        Mmax = max(np.abs(feature_importance))
        selected_list = []
        for i in range(int(x.shape[1]*self.variable_size)):
            j = int(len(feature_importance) * np.random.random())#随机取个数
            m = Mmax * np.random.random()#随机判断阈值m
            goon = True
            while goon:
                m = Mmax * np.random.random()
                j = int(len(feature_importance) * np.random.random())
                if np.abs(feature_importance[j]) < m and j not in selected_list:
                    selected_list.append(j)
                    goon = False
        return selected_list

    def choose_variable(self,x,feature_importance = None):
        
        if feature_importance is None:
            new_var = random.sample(range(x.shape[1]),int(x.shape[1]*self.variable_size))
        else:
            #采用Lahiri法进行带权重采样
            new_var = self.lahiri(x,feature_importance)
        return x[:,new_var],new_var

    def sample(self,x,y,feature_importance=None):
        sample_list = []
        var_list = []
        for i in range(self.n_models):
            x_ = x.copy()
            y_ = y.copy()
            x_,y_ = self.bootstrap(x_,y_)
            x_,new_var = self.choose_variable(x_,feature_importance=feature_importance)
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
        return feature_importance['feature_importance'].values

    def second_fit(self,x,y,feature_importance,method = 'AdaptiveLasso'):
        '''
        第二次自助取样，根据特征权重来选择
        '''
        f_importance = pd.DataFrame()
        f_importance['index'] = [col for col in range(x.shape[1])]
        sample_list,self.variables = self.sample(x,y,feature_importance=feature_importance)
        for i in range(len(sample_list)):
            sample = sample_list[i]
            variables = self.variables[i]
            if method == 'AdaptiveLasso':
                clf = AdaptiveLasso()
                clf.fit(sample['x'],sample['y'])
                new_feature_importances = pd.DataFrame({'index':variables,'weight%d' % i:clf.weights})
            elif method == 'Lasso':
                clf = Lasso(fit_intercept=False)
                clf.fit(sample['x'],sample['y'])
                new_feature_importances = pd.DataFrame({'index':variables,'weight%d' % i:clf.coef_})
            else:
                raise Exception("Wrong Method!!!!")
                
            f_importance = pd.merge(f_importance,new_feature_importances,how='left')
            self.models.append(clf)
        f_importance = f_importance.fillna(0)
        f_importance['feature_importance'] = np.mean(f_importance.drop('index',axis=1),axis=1)
        return f_importance['feature_importance'].values
    
    def fit(self,x,y):
        feature_importance = self.first_fit(x,y)
        
        self.coef_ = self.second_fit(x,y,feature_importance,method = 'Lasso')
        print(self.coef_)
    def predict(self,x):
        return np.dot(x,self.coef_)

def read_boston_data():
    boston = load_boston()
    x = np.array(boston.data)
    y = np.array(boston.target)

    return x,y

if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    clf = RandomLasso(n_models=50,variable_size=0.9)
    x,y = read_boston_data()
    #x,y = make_regression(n_samples=100,n_features = 15)
    clf.fit(x,y)
    y_pred = clf.predict(x)
    clf2 = Lasso()
    clf2.fit(x,y)
    y_pred2 = clf2.predict(x)
    clf3 = AdaptiveLasso()
    clf3.fit(x,y)
    y_pred3 = clf3.predict(x)

    print('random',np.mean(np.abs(y_pred-y)))
    print('lasso',np.mean(np.abs(y_pred2-y)))
    print('ada',np.mean(np.abs(y_pred3-y)))
    plt.plot(range(len(y)),y_pred3,color = 'pink')
    plt.plot(range(len(y)),y_pred2,color = 'orange')
    plt.plot(range(len(y)),y_pred,color = 'red')
    plt.plot(range(len(y)),y,color = 'green')
    plt.show()