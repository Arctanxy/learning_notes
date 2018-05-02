import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split ,cross_val_score
from sklearn import ensemble,tree,linear_model,svm,neighbors,grid_search
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.neural_network import MLPRegressor

#读取数据
def read_x_y():
    train = pd.read_csv("E:\kaggle\houseprice\data\Mtrain.csv")
    test = pd.read_csv("E:\kaggle\houseprice\data\Mtest.csv")
    del test['SalePrice']
    #提取训练集中的xy
    y = train.pop('SalePrice')
    y = np.log(y)
    #y = np.log(y)#log有风险，因为正则化的过程中会出现1，log之后为0，而正则化得到的0，log过程中会报错
    del train['Id']
    testId = test.pop('Id')
    x = train
    return x,y,test,testId

#评分函数
def get_score(y_pred,y_true):
    print('R2:{}'.format(r2_score(y_pred,y_true)))
    print('RMSE:{}'.format(np.sqrt(mean_squared_error(y_pred,y_true))))

def train_test(estimator,x_train,x_test,y_train,y_test):
    train_pred = estimator.predict(x_train)
    print(estimator)
    #训练得分
    get_score(train_pred,y_train)
    test_pred = estimator.predict(x_test)
    print("Test")
    #测试得分
    get_score(test_pred,y_test)



if __name__ == '__main__':
    x,y,test_data,test_Id = read_x_y()
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=200)
    #模型1,线性回归
    ENSTest = linear_model.ElasticNetCV(alphas=[0.0001,0.0005,0.001,0.01,0.1,1,10],
                                        l1_ratio=[0.01,0.1,0.5,0.9,0.99],
                                        max_iter=5000).fit(x_train,y_train)
    train_test(ENSTest,x_train,x_test,y_train,y_test)
    #交叉检验
    scores1 = cross_val_score(ENSTest,x,y,cv=5)
    print('Accuracy:%0.4f (+/- %0.4f)' % (scores1.mean(),scores1.std()*2))
    #模型2，GradientBoostingRegressor
    GBest =ensemble.GradientBoostingRegressor(n_estimators = 5000,learning_rate =0.05,
                                              max_depth = 3,max_features = 'sqrt',
                                              min_samples_leaf=15,min_samples_split = 10,
                                             loss='huber').fit(x_train,y_train)
    train_test(GBest,x_train,x_test,y_train,y_test)
    #交叉检验
    scores2 = cross_val_score(GBest,x,y,cv=5)
    print('Accuracy:%0.4f (+/- %0.4f)' % (scores2.mean(),scores2.std()*2))

    '''当y中数值为float时，LogisticRegression()会报错'''
    #逻辑回归
    #LogReg = linear_model.LogisticRegression().fit(x_train,y_train)
    #train_test(LogReg,x_train,x_test,y_train,y_test)
    #交互验证
    #scores3 = cross_val_score(LogReg,x,y,cv=5)
    #print('Accuracy:%0.4f (+/- %0.4f)' % (scores3.mean(),scores3.std()*2))

    #支持向量机
    parameter_spaceSVR ={
        'C':[0.01,0.1,1,10,100],
        'gamma':[1e-4,1e-3,1e-2,0.1,1,10]
    }
    S = svm.SVR()
    SR = grid_search.GridSearchCV(S,parameter_spaceSVR)
    SR.fit(x_train,y_train)
    print('Best params:%r'% SR.best_params_)
    train_test(SR,x_train,x_test,y_train,y_test)
    #交叉验证
    scores3 = cross_val_score(SR,x,y,cv=5)
    print('Accuracy:%0.4f (+/- %0.4f)' % (scores3.mean(),scores3.std()*2))

    #随机森林
    parameter_sapceRF = {
        "max_features":[0.6,0.4,0.5],
        "n_estimators":[100,200,300],
        "min_samples_leaf":[1,2],
    }
    #使用GridSearchCV进行自动调参
    RForest = ensemble.RandomForestRegressor(random_state=14)
    RFgrid = grid_search.GridSearchCV(RForest,parameter_sapceRF)
    RFgrid.fit(x_train,y_train)
    print('Best params:%r' % RFgrid.best_params_)
    #交叉验证
    train_test(RFgrid,x_train,x_test,y_train,y_test)
    scores4 = cross_val_score(RFgrid,x,y,cv=5)#这一步速度极慢
    print('Accuracy:%0.4f (+/- %0.4f)' % (scores4.mean(),scores4.std()*2))

    #贝叶斯回归——表现较好，暂不进行调参
    Byes = linear_model.BayesianRidge().fit(x_train,y_train)
    train_test(Byes,x_train,x_test,y_train,y_test)
    #交叉验证
    scores5 = cross_val_score(Byes,x,y,cv=5)
    print('Accuracy:%0.4f (+/- %0.4f)' % (scores5.mean(),scores5.std()*2))

    #最近邻
    parameter_spaceKNN = {
        'n_neighbors':list(range(5,15)),
        'algorithm':['auto','ball_tree','kd_tree'],
        'weights':['uniform','distance'],
        'leaf_size':[1,2],
    }
    KN = neighbors.KNeighborsRegressor()
    KNN = grid_search.GridSearchCV(KN,parameter_spaceKNN)
    KNN.fit(x_train,y_train)
    print('Best params:%r' % KNN.best_params_)
    train_test(KNN,x_train,x_test,y_train,y_test)
    #交叉验证
    scores6 = cross_val_score(KNN,x,y,cv=5)
    print('Accuracy:%0.4f (+/- %0.4f)' % (scores6.mean(),scores6.std()*2))

    #神经网络
    MLP = MLPRegressor(solver='lbfgs',hidden_layer_sizes=(20,20,20),random_state=1).fit(x_train,y_train)
    train_test(MLP,x_train,x_test,y_train,y_test)
    #交叉验证
    scores7 = cross_val_score(MLP,x,y,cv=5)
    print('Accuracy:%0.4f (+/- %0.4f)' % (scores7.mean(),scores7.std()*2))


    #简单融合
    final = (np.exp(ENSTest.predict(test_data))+2*np.exp(GBest.predict(test_data))+
             np.exp(SR.predict(test_data))+np.exp(RFgrid.predict(test_data))+
             np.exp(Byes.predict(test_data))+np.exp(KNN.predict(test_data))+np.exp(MLP.predict(test_data)))/8
    pd.DataFrame({'Id':test_Id,'SalePrice':final}).to_csv('E:/kaggle/houseprice/data/result.csv',index = False)






