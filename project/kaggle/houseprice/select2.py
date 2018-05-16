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
    scores = cross_val_score(clf,x,y,scoring='neg_mean_squared_error',cv=cv)
    return np.mean([np.sqrt((-1)*score) for score in scores])

# 后向cv
train_data = pd.read_csv(PATH + 'step_train.csv')
test_data = pd.read_csv(PATH + 'step_test.csv')
clf = RidgeCV(alphas=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1])
x = train_data.drop(['SalePrice','Id'],axis=1)
y = np.log(train_data['SalePrice'])
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
x['SalePrice'] = y
x.to_csv(PATH + 'selected_train.csv')
test = test_data[x.columns]
test.to_csv(PATH + 'selected_test.csv')
