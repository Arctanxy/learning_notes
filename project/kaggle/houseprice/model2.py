from sklearn.linear_model import Ridge,Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.base import clone
from xgboost import XGBRegressor
import pandas as pd 
import numpy as np 
from feature2 import PATH
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore") # 忽略warning，但是似乎没什么用

def load_data(method='backward'):
    if method == 'backward':
        test = pd.read_csv(PATH + 'selected_test.csv')
        train = pd.read_csv(PATH + 'selected_train.csv')
    else:
        test = pd.read_csv(PATH + 'selected_test2.csv')
        train = pd.read_csv(PATH + 'selected_train2.csv')
    x = train.drop(['SalePrice','Id'],axis=1)
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
    predictions = np.zeros((test.shape[0],len(models)))
    for i,m in enumerate(models):
        yp = m.predict(test.drop('Id',axis=1))
        # predictions.append(yp)
        predictions[:,i] = yp
    return np.squeeze(np.dot(predictions,weights))

class stack_model:
    '''使用KFold的方式将数据集划分为5个部分，使用每个basemodel训练五次，
    再预测五次，合并得到一个predict_price，作为mergemodel中的自变量'''
    def __init__(self,base_models,merge_model,n_folds = 5):
        self.base_models = base_models
        self.merge_model = merge_model
        self.n_folds = n_folds

    def fit(self,x,y):
        self.fitted_models = [list() for x in self.base_models] # 用于存储训练之后的模型
        kfold = KFold(n_splits = self.n_folds,shuffle = True)
        out_of_fold_predictions = np.zeros((x.shape[0],len(self.base_models)))
        for i,model in enumerate(self.base_models):
            for train_index,valid_index in kfold.split(x,y):
                instance = clone(model)
                instance.fit(x.iloc[train_index],y.iloc[train_index])
                self.fitted_models[i].append(instance)
                y_pred = instance.predict(x.iloc[valid_index])
                out_of_fold_predictions[valid_index,i] = y_pred
        self.merge_model.fit(out_of_fold_predictions,y)
        return self
    
    def predict(self,x):
        merge_features = np.column_stack([
            np.column_stack([
                model.predict(x) for model in models
            ]).mean(axis=1) for models in self.fitted_models
        ])
        return self.merge_model.predict(merge_features)


if __name__ == "__main__":
    test,x,y = load_data()
    raw_test = pd.read_csv(PATH + 'test.csv')  #原始数据中的GrLivArea字段已经被删掉了，￣□￣｜｜
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
    
    '''rf = search_model(RandomForestRegressor(),x,y,params={
        'n_estimators':[300,500,800],
        'max_features':[0.5,'sqrt',0.8],
        'min_samples_leaf':[2,3,4],
        'n_jobs':[-1],
        'max_depth':[3,5,7,9,11]
    })'''   # {'max_depth': 11, 'max_features': 0.5, 'min_samples_leaf': 2, 'n_estimators': 800, 'n_jobs': -1}
            #  best_score 0.13587390392097842
    
    
    '''krr = search_model(KernelRidge(),x,y,params={
        'alpha':[1e-4,1e-3,1e-2,1e-1,1e0,1e1],
        'kernel':['linear','polynomial','rbf'],
        'degree':[2,3,4],
    })'''   # best_params: {'alpha': 1.0, 'degree': 2, 'kernel': 'linear'}
            # best_score 0.13702834086153534
    

    '''gbd = search_model(GradientBoostingRegressor(),x,y,params = {
        'loss':['ls', 'lad', 'huber', 'quantile'],
        'learning_rate':[1e-4,1e-3,1e-2,1e-1],
        'n_estimators':[100,200,400],
        'criterion':['mse'],
        'max_features':['sqrt','log2']
    })'''  # best_params: {'criterion': 'mse', 'learning_rate': 0.1, 'loss': 'huber', 'max_features': 'sqrt', 'n_estimators': 400}
            # best_score 0.12006154048626891
    
            
    rid = Ridge(alpha=1,fit_intercept=True,normalize=False)
    las = Lasso(alpha=1e-5,fit_intercept=True,normalize=True,max_iter=300)
    xg = XGBRegressor(learning_rate=0.1,max_depth=2,n_estimators=500,reg_alpha=0.2,reg_lambda=0.7)
    rf = RandomForestRegressor(max_depth=11,max_features=0.5,min_samples_leaf=2,n_estimators=800,n_jobs=(-1))
    krr = KernelRidge(alpha=1.0,degree=2,kernel='linear')
    gbd = GradientBoostingRegressor(criterion='mse',learning_rate=0.1,loss='huber',max_features='sqrt',n_estimators=400)

    # models = [rid,las,xg,rf,krr,gbd]
    models = [rid,las,xg,krr,gbd]
    s_model = stack_model(models[:len(models)-1],models[len(models)-1])
    for i in tqdm(range(len(models))):
        models[i].fit(x,y)
    y_predict = voting_predict(models,test)
    submission = pd.DataFrame({'Id':test['Id'],'SalePrice':np.exp(y_predict) * raw_test['GrLivArea']})
    # submission.to_csv(PATH + 'backward_result.csv',index=False)
    submission.to_csv(PATH+'forward_result.csv',index=False)
    s_model.fit(x,y)
    y_predict2 = s_model.predict(test.drop('Id',axis=1))
    submission2 = pd.DataFrame({'Id':test['Id'],'SalePrice':np.exp(y_predict2)  * raw_test['GrLivArea']})
    # submission2.to_csv(PATH + 'backward_result2.csv',index=False)
    submission2.to_csv(PATH + 'forward_result2.csv',index=False)