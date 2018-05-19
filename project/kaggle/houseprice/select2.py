from sklearn.linear_model import RidgeCV,LassoCV
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.grid_search import GridSearchCV
from feature2 import PATH
import pandas as pd 
import numpy as np 
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def check(x,y,clf=RidgeCV(alphas=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1]),cv=10):
    if len(x.shape) == 1 :
        scores = cross_val_score(clf,x.reshape(-1,1),y,scoring='neg_mean_squared_error',cv=cv)
    else:
        scores = cross_val_score(clf,x,y,scoring='neg_mean_squared_error',cv=cv)
    return np.mean([np.sqrt((-1)*score) for score in scores])

def backward_cv(train_data,clf = RidgeCV(alphas=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1])):
    '''
    逐步删除特征,运算时间较长，clf尽量选择简单模型
    '''
    x = train_data.drop(['SalePrice','AVG_PRICE','Id'],axis=1)
    y = np.log(train_data['AVG_PRICE'])
    best_score = check(x,y)
    dropped_col = []
    for col in tqdm(x.columns):
        score = check(x.drop(col,axis=1),y)
        if score <= best_score:
            x = x.drop(col,axis=1)
            best_score = score
            print(score)
            dropped_col.append(col)
        else:
            pass
    print(x.shape,best_score)
    return x.columns

def forward_cv(train_data,clf = RidgeCV(alphas=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1])):
    '''
    逐步增加特征
    '''
    x = train_data.drop(['SalePrice','AVG_PRICE','Id'],axis=1)
    y = np.log(train_data['AVG_PRICE'])
    best_score = np.inf
    # 先寻找最好的单特征
    best_col = ""
    for col in tqdm(x.columns):
        score = check(x[col].reshape(-1,1),y)
        if score < best_score:
            best_score = score
            best_col = col
            print(score)
    x_new = pd.DataFrame({
        best_col:x[best_col]
    })
    print('===best_col==',best_col)
    for col in tqdm(x.drop(best_col,axis=1).columns):
        x_new[col] = x[col] # 这一列莫名其妙地加到了行上面
        score = check(x_new,y)
        if score < best_score:
            best_score = score
            print(col,score)
        elif len(x_new.shape) > 1:
            x_new = x_new.drop(col,axis=1)
    return x_new.columns

train_data = pd.read_csv(PATH + 'step_train.csv')

y = train_data['AVG_PRICE']
x = train_data.drop(['SalePrice','AVG_PRICE','Id'],axis=1)

del train_data['SaleTime']
test_data = pd.read_csv(PATH + 'step_test.csv')
del test_data['SaleTime']
# print(np.where(np.isnan(train_data)),'\n')
# print(np.where(np.isnan(test_data)))
clf = RidgeCV(alphas=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1])
tr_ids = train_data.Id
te_ids = test_data.Id

# selected_index = forward_cv(train_data)
selected_index = backward_cv(train_data)
test_data = test_data[selected_index]
x = x[selected_index]

x['SalePrice'] = y
x['Id'] = tr_ids
x.to_csv(PATH + 'selected_train2.csv',index=False)# 1为backward;2为forward
test_data['Id'] = te_ids
test_data.to_csv(PATH + 'selected_test2.csv',index=False)
