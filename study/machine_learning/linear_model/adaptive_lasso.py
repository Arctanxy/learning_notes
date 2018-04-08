'''
Adaptive Lasso实现
也被成为迭代加权L1范数回归
'''

from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
import numpy as np 



class AdaptiveLasso(object):
    def __init__(self,alpha = 1e-5,iterations = 5):
        self.alpha = alpha
        self.iterations = iterations

    def fit(self,x,y):
        g = lambda w: np.sqrt(np.abs(w))
        gprime = lambda w:1./(2.*g(w) + np.finfo(float).eps)
        
        n_samples,n_features = x.shape

        p_obj = lambda w:1./(2.*n_samples) * np.sum((y-np.dot(x,w))**2) + self.alpha * np.sum(g(w))#误差项
        
        weights = np.ones(n_features)

        for i in range(self.iterations):
            x_w = x/weights[np.newaxis,:]
            clf = Lasso(alpha=self.alpha,fit_intercept=False)
            clf.fit(x_w,y)
            coef_ = clf.coef_/weights
            weights = gprime(coef_)
            print(p_obj(coef_))

    def predict(self,x):
        return np.dot(x,w)

if __name__ == '__main__':
    x,y = make_regression(n_samples=100,n_features = 10)

    
    Ada_clf = AdaptiveLasso()
    Ada_clf.fit(x,y)
