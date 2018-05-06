from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso,ElasticNet,LassoCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from features import PATH
import numpy as np 
import pandas as pd
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")


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

if __name__ == "__main__":
    train_data = pd.read_csv(PATH + 'new_train.csv')
    test_data = pd.read_csv(PATH + 'new_test.csv')


    las = Lasso(alpha=0.0005) # alpha值通过LassoCV()测试得到
    xg = xgb.XGBRegressor(learning_rate=0.1,max_depth=2,n_estimators=500,reg_alpha=0.3,reg_lambda=0.6) # 通过GridSearchCV测试得到
    rf = RandomForestRegressor()
    # rf = RandomForestRegressor(criterion='mse',max_depth=11,max_features=0.5,min_samples_leaf=3,n_estimators=800,n_jobs=-1)
    krr = KernelRidge()
    # For xgboost
    '''params = {
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
    search_model(rf,params,train_data)'''

    s1 = make_prediction(las,train_data,test_data)
    s2 = make_prediction(xg,train_data,test_data)
    s3 = make_prediction(krr,train_data,test_data)
    s = pd.DataFrame()
    s['Id'] = s1['Id']
    s['SalePrice'] = (s1['SalePrice'] + s2['SalePrice'] + s3['SalePrice'])/3
    s.to_csv(PATH + 'my_submission.csv',index=False)
    