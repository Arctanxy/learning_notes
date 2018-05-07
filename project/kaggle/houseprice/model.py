from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso,ElasticNet,LassoCV,RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest # 用于选择重要的特征
from sklearn.feature_selection import chi2
from features import PATH
import numpy as np 
import pandas as pd
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore") # 忽略warning，但是似乎没什么用


def select(clf,k,train_data):
    x = train_data.drop(['SalePrice','SaleTime','Id'],axis=1)
    y = train_data['SalePrice']
    x_new = SelectKBest(chi2,k=k).fit_transform(x,y)
    score = np.mean(cross_val_score(clf,x_new,y,cv=5))
    return score


def make_prediction(clf,train_data,test_data):
    x = train_data.drop(['SalePrice','SaleTime','Id'],axis=1)
    y = train_data['SalePrice']
    clf.fit(x,y)
    print(str(clf),np.mean(cross_val_score(clf,x,y,cv=5)))

    x_test = test_data.drop(['SaleTime','Id','SalePrice'],axis=1)
    y_pred = np.exp(clf.predict(x_test))
    submission = pd.DataFrame()
    submission['Id'] =  test_data['Id']
    submission['SalePrice'] = y_pred
    return submission

def search_model(clf,params,train_data):
    x = train_data.drop(['SalePrice','SaleTime','Id'],axis=1)
    y = train_data['SalePrice']
    new_clf = GridSearchCV(clf,params,scoring='neg_mean_squared_error')
    new_clf.fit(x,y)
    print(str(clf),new_clf.best_score_,new_clf.best_params_)

class blend_model:
    def __init__(self,base_model,vote_model):
        self.base_model = base_model
        self.vote_model = vote_model
    
    def fit_(self,x,y):
        predictions = []
        for i in range(len(self.base_model)):
            self.base_model[i].fit(x,y)
            pred = self.base_model[i].predict(x)
            predictions.append(pred)
        x_new = np.reshape(predictions,(x.shape[0],len(predictions)))
        self.vote_model.fit(x_new,y)
    
    def predict_(self,x):
        predictions = []
        for i in range(len(self.base_model)):
            pred = self.base_model[i].predict(x)
            predictions.append(pred)
        x_new = np.reshape(predictions,(x.shape[0],len(predictions)))
        result = self.vote_model.predict(x_new)
        return result




if __name__ == "__main__":
    train_data = pd.read_csv(PATH + 'new_train.csv')
    x = train_data.drop(['SalePrice','SaleTime','Id'],axis=1)
    y = train_data['SalePrice']
    test_data = pd.read_csv(PATH + 'new_test.csv')


    las = Lasso(alpha=0.0005) # alpha值通过LassoCV()测试得到
    xg = xgb.XGBRegressor(learning_rate=0.1,max_depth=2,n_estimators=500,reg_alpha=0.2,reg_lambda=0.4) # 通过GridSearchCV测试得到
    # rf = RandomForestRegressor()
    rf = RandomForestRegressor(criterion='mse',max_depth=11,max_features=0.5,min_samples_leaf=3,n_estimators=800,n_jobs=-1)
    krr = KernelRidge(alpha = 0.1,degree=2,gamma=0.01,kernel='linear')
    rid = RidgeCV(alphas=[1e-4,1e-3,1e-2,0.1])


    # For xgboost
    '''
    params = {
        'learning_rate':[0.1],
        'max_depth':[2],
        'n_estimators':[500],
        'reg_alpha':[0.2,0.3,0.4,0.5,0.6],
        'reg_lambda':[0.2,0.3,0.4,0.5,0.6,0.7]
    }'''

    # For RandomForest
    '''
    params = {
        'n_estimators':[800],
        'criterion':['mse'],
        'max_features':[0.5],
        'min_samples_leaf':[3],
        'n_jobs':[-1],
        'max_depth':[11]
    }
    '''
    # For KernelRidge()
    '''
    params = {
        'alpha':[1e-4,1e-3,1e-2,0.1],
        'gamma':[1e-2,1e-3,0.1],
        'kernel':['linear','laplacian','polynomial','sigmoid'],
        'degree':[2,3],
    }
    search_model(krr,params,train_data)'''

    s1 = make_prediction(las,train_data,test_data)
    s2 = make_prediction(xg,train_data,test_data)
    s3 = make_prediction(krr,train_data,test_data)
    
    s = pd.DataFrame()
    s['Id'] = s1['Id']
    s['SalePrice'] = (2*s1['SalePrice'] + s2['SalePrice'] + s3['SalePrice'])/4 # + s4['SalePrice'])/4
    s.to_csv(PATH + 'my_submission.csv',index=False)


    # 使用blend_model效果并不好，可以说是出乎意料的差，可能是代码有问题，待检查
    '''bm = blend_model([las,xg,krr],rid)
    bm.fit_(x,y)
    # 测试blend_model 的效果
    y_p = bm.predict_(x)
    y_p = np.exp(y_p)
    y = np.exp(y)
    print(np.mean(np.abs(y_p-y)/y))

    # 使用blend_model预测test
    test_data = test_data.drop(['SalePrice','SaleTime','Id'],axis=1)    
    y_pred = bm.predict_(test_data)
    
    s['SalePrice'] = np.exp(y_pred)
    s.to_csv(PATH + 'my_submission2.csv',index=False)'''

    
