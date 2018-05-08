from sklearn.ensemble import RandomForestRegressor #用于特征筛选
from sklearn.linear_model import Lasso,ElasticNet,LassoCV,RidgeCV,LinearRegression
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
warnings.filterwarnings("ignore") # 忽略warning，但是似乎没什么用


def select(clf,k,x,y):
    x_new = SelectKBest(chi2,k=k).fit_transform(x,y)
    score = np.mean(cross_val_score(clf,x_new,y,cv=5))
    return score

def select_variables(x,y): # ,k):
    '''
    选取重要性排名在前k个的因素
    '''
    rf = RandomForestRegressor()
    rf.fit(x,y)
    cols = x.columns
    f_imp = dict(zip(cols,rf.feature_importances_))
    sorted_f = sorted(f_imp.items(),key=lambda x:x[1],reverse = True) # 按重要性从大到小排列
    
    return sorted_f # 直接返回排序后的元组，方便修改k值
    # return [s[0] for s in sorted_f[:k]] # 返回前k特征名称


def make_prediction(clf,x,y,x_test,test_data):
    clf.fit(x,y)
    print(str(clf),np.mean(cross_val_score(clf,x,y,cv=5)))
    y_pred = np.exp(clf.predict(x_test))
    submission = pd.DataFrame()
    submission['Id'] =  test_data['Id']
    submission['SalePrice'] = y_pred
    return submission

def search_model(clf,params,x,y):
    new_clf = GridSearchCV(clf,params,scoring='neg_mean_squared_error')
    new_clf.fit(x,y)
    print(str(clf),new_clf.best_score_,new_clf.best_params_)

class blend_model:
    '''
    一种模型融合方法，但在此处应用效果并不好
    '''
    def __init__(self,base_model,vote_model):
        self.base_model = base_model
        self.vote_model = vote_model
    
    def fit_(self,x,y):
        predictions = []
        for i in range(len(self.base_model)):
            self.base_model[i].fit(x,y)
            pred = self.base_model[i].predict(x)
            predictions.append(np.exp(pred)) # 将预测值还原
        # x_new = np.reshape(predictions,(x.shape[0],len(predictions))) # reshape 错误，把数据打乱了
        x_new = np.transpose(np.array(predictions))
        self.vote_model.fit(x_new,np.exp(y))
    
    def predict_(self,x):
        predictions = []
        for i in range(len(self.base_model)):
            pred = self.base_model[i].predict(x)
            predictions.append(np.exp(pred)) # 将预测值还原
        x_new = np.transpose(np.array(predictions))
        result = self.vote_model.predict(x_new)
        return result

def build_model(x,y,test_data,k):
    print('==筛选特征中==')
    selected_variables = [s[0] for s in select_variables(x,y)[:k]]
    x = x[selected_variables]
    x_test = test_data[selected_variables]

    las = Lasso(alpha=0.0005) # alpha值通过LassoCV()测试得到
    xg = xgb.XGBRegressor(learning_rate=0.1,max_depth=2,n_estimators=500,reg_alpha=0.2,reg_lambda=0.4) # 通过GridSearchCV测试得到
    rf = RandomForestRegressor(criterion='mse',max_depth=11,max_features=0.5,min_samples_leaf=3,n_estimators=800,n_jobs=-1)
    krr = KernelRidge(alpha = 0.1,degree=2,gamma=0.01,kernel='linear')
    rid = RidgeCV(alphas=[1e-4,1e-3,1e-2,0.1])
    lin = LinearRegression()


    print('==训练模型中==')
    s1 = make_prediction(las,x,y,x_test,test_data) # 参数中使用test_data只为了提供Id
    s2 = make_prediction(xg,x,y,x_test,test_data)
    s3 = make_prediction(krr,x,y,x_test,test_data)
    
    s = pd.DataFrame()
    s['Id'] = s1['Id']
    s['SalePrice'] = (2*s1['SalePrice'] + s2['SalePrice'] + s3['SalePrice'])/4 # + s4['SalePrice'])/4
    s.to_csv(PATH + 'my_submission_%d.csv' % k,index=False)
    
    # 重新定义三个模型
    las1 = Lasso(alpha=0.0005) # alpha值通过LassoCV()测试得到
    xg1 = xgb.XGBRegressor(learning_rate=0.1,max_depth=2,n_estimators=500,reg_alpha=0.2,reg_lambda=0.4) # 通过GridSearchCV测试得到
    krr1 = KernelRidge(alpha = 0.1,degree=2,gamma=0.01,kernel='linear')
    rid = RidgeCV(alphas=[1e-4,1e-3,1e-2,0.1,1])

    print('==尝试模型融合==')
    # y = np.exp(y) # 将y还原
    # 使用blend_model效果并不好，可以说是出乎意料的差，可能是代码有问题，待检查
    bm = blend_model([las1,xg1,krr1],rid)
    bm.fit_(x,y)
    # 测试blend_model 的效果
    y_p = bm.predict_(x)
    print(np.mean(np.abs(y_p-np.exp(y))/np.exp(y)))

    # 使用blend_model预测test
    y_pred = bm.predict_(x_test)
    
    s['SalePrice'] = y_pred
    s.to_csv(PATH + 'my_submission2_%d.csv' % k,index=False)



if __name__ == "__main__":
    train_data = pd.read_csv(PATH + 'new_train.csv')
    x = train_data.drop(['SalePrice','SaleTime','Id'],axis=1)
    print(x.shape)
    y = train_data['SalePrice']
    test_data = pd.read_csv(PATH + 'new_test.csv')

    # 选取不同的特征，得到新的模型，事实证明，特征越多效果越好
    for k in [100,150,200,240]:
        print("取前%d个元素" % k)
        build_model(x,y,test_data,k)


    





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