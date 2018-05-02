import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge,BayesianRidge
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt 
from xgboost import XGBRegressor

def accuracy(y_pred,y_test):
    return np.mean([abs(np.array(y_pred)[i]-np.array(y_test)[i])/np.array(y_test)[i] for i in range(len(y_pred))])

train = pd.read_csv('./data/Mtrain.csv')
train = train.drop(['Id'],axis=1)

test = pd.read_csv('./data/Mtest.csv')
Id = test['Id']
test = test.drop(['SalePrice','Id'],axis=1).astype('float64')

x = train.drop('SalePrice',axis=1).astype('float64')
y = train['SalePrice'].astype('float64')

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state = 100)

clf = XGBRegressor()#criterion='mae'效果反而差一点
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
                
print('rf',accuracy(y_pred,y_test))


plt.plot(y_test,y_test,color = 'r',alpha = 0.6)
plt.scatter(y_test,y_pred,color = 'g',alpha = 0.6)

plt.show()
submission = pd.DataFrame()
submission['Id'] = Id
submission['SalePrice'] = clf.predict(test)
submission.to_csv('./data/result.csv',index=False)