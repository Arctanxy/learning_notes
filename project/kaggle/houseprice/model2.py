from sklearn.linear_model import Ridge,Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import pandas as pd 
import numpy as np 
from feature2 import PATH
import warnings
warnings.filterwarnings("ignore") # 忽略warning，但是似乎没什么用

def load_data():
    test = pd.read_csv(PATH + 'selected_test.csv')
    train = pd.read_csv(PATH + 'selected_train.csv')
    x = train.drop('SalePrice',axis=1)
    y = train['SalePrice']
    return test,x,y

def search_model(clf,x,y,params):
    '''网格调参'''
    model = GridSearchCV(clf,param_grid = params,cv=10,scoring= 'neg_mean_squared_error')
    model.fit(x,y)
    print('best_params:',model.best_params_,'\n','best_score',np.sqrt((-1)*model.best_score_))
    return model

def voting_predict(models,test,weights='auto'):
    '''表决结果'''
    if weights == 'auto':
        weights = [1/len(models) for i in range(len(models))]
    weights = np.array(weights).reshape(-1,1)
    predictions = []
    for m in models:
        yp = m.predict(test).reshape(-1,1)
        predictions.append(yp)
    predictions = np.transpose(predictions)
    return predictions*weights

if __name__ == "__main__":
    test,x,y = load_data()
    '''rid = search_model(Ridge(),x,y,params = {
        'alpha': [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1],
        'fit_intercept': [True,False],
        'normalize': [True,False],
    })'''    #    {'alpha': 1, 'fit_intercept': True, 'normalize': False}
            #  best_score 0.12629291057923642
    '''las = search_model(Lasso(),x,y,params = {
        'alpha':[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1],
        'fit_intercept': [True,False],
        'normalize': [True,False],
        'max_iter':[100,300,500]
    })'''    #    best_params: {'alpha': 1e-05, 'fit_intercept': True, 'max_iter': 300, 'normalize': True}
             #   best_score 0.12698849132599
    '''xg = search_model(XGBRegressor(),x,y,params = {
        'learning_rate':[0.1],
        'max_depth':[2],
        'n_estimators':[500],
        'reg_alpha':[0.2,0.3,0.4,0.5,0.6],
        'reg_lambda':[0.2,0.3,0.4,0.5,0.6,0.7]
    })'''   #   best_params: {'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 500, 'reg_alpha': 0.2, 'reg_lambda': 0.7}
            #   best_score 0.12042016330378752
    rf = search_model(RandomForestRegressor(),x,y,params={
        'n_estimators':[300,500,800],
        'max_features':[0.5,'sqrt',0.8],
        'min_samples_leaf':[2,3,4],
        'n_jobs':[-1],
        'max_depth':[3,5,7,9,11]
    })
