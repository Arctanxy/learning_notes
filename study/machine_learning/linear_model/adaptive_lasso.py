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
        self.weights = None

    def fit(self,x,y):
        g = lambda w: np.sqrt(np.abs(w))
        gprime = lambda w:1./(2.*g(w) + np.finfo(float).eps)
        
        n_samples,n_features = x.shape

        p_obj = lambda w:1./(2.*n_samples) * np.sum((y-np.dot(x,w))**2) + self.alpha * np.sum(g(w))#误差项
        
        self.weights = np.ones(n_features)

        for i in range(self.iterations):
            x_w = x/self.weights[np.newaxis,:]
            clf = Lasso(alpha=self.alpha,fit_intercept=False)
            clf.fit(x_w,y)
            coef_ = clf.coef_/self.weights
            self.weights = gprime(coef_)
            if i == self.iterations - 1:
                self.weights = coef_

    def predict(self,x):
        self.weights = np.reshape(self.weights,(len(self.weights),1))
        return np.dot(x,self.weights)

if __name__ == '__main__':
    import matplotlib.pyplot as plt 
    x,y = make_regression(n_samples=10,n_features = 10)

    
    Ada_clf = AdaptiveLasso(iterations=100)
    Ada_clf.fit(x,y)
    y_pred = Ada_clf.predict(x)
    plt.plot(range(len(y)),y_pred,color = 'red')
    plt.plot(range(len(y)),y,color = 'green')
    plt.show()
