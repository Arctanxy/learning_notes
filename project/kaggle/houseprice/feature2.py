# 特征工程
# dalalaa
import pandas as pd 
import numpy as np 
from tqdm import tqdm
import warnings
from features1 import Add_reference_price,combine_time
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

# 读取数据

PATH = 'H:/learning_notes/project/kaggle/houseprice/data/'
# PATH = 'C:/Users/Dl/Documents/GitHub/learning_notes/project/kaggle/houseprice/data/'
def load_data():
    train_data = pd.read_csv(PATH + 'train.csv')
    train_data['AVG_PRICE'] = train_data['SalePrice']/train_data['GrLivArea'] # 价格使用每平方英尺的均价进行计算
    test_data = pd.read_csv(PATH + 'test.csv')
    train_data = combine_time(train_data)
    test_data = combine_time(test_data)
    data = pd.concat([train_data,test_data],axis=0)
    data = Add_reference_price(data,train_data)
    return train_data,test_data,data

def manage_data(data,train_data,test_data):

    # 等级替换
    '''data = data.replace({
        'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1
    })'''
    
    # 处理缺失值
        # 区域因素
    data['MSZoning'] = data['MSZoning'].fillna(data['MSZoning'].mode().iloc[0])
        # 交通地形因素
    for f in ['Street','Alley','LandContour','LandSlope','Condition1','Condition2']:
        data[f] = data[f].fillna(data[f].mode().iloc[0])
    data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].mean())
        # 房屋总体特征
    for f in ['MasVnrType','MasVnrArea','Exterior1st','Exterior2nd','Functional']:
        data[f] = data[f].fillna(data[f].mode().iloc[0])
        # 房屋内部配置
    for f in ['BsmtQual','BsmtCond','BsmtFinSF1','BsmtFinSF2','BsmtFullBath','BsmtUnfSF','BsmtHalfBath',
            'GarageQual','GarageCond','PoolQC','KitchenQual','GarageArea','GarageCars']:
        data[f] = data[f].fillna(0)
    for f in ['BsmtExposure','BsmtFinType1','BsmtFinType2','GarageType','GarageYrBlt','GarageFinish','Fence','MiscFeature']:
        data[f] = data[f].fillna('None')
    data['TotalBsmtSF'] = data['TotalBsmtSF'].fillna(data['TotalBsmtSF'].mean())
    data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode().iloc[0])
    data['FireplaceQu'] = data['FireplaceQu'].fillna(0.0)
        # 销售信息
    data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode().iloc[0])
        # 其他
    data['Utilities'] = data['Utilities'].fillna(data['Utilities'].mode().iloc[0])

    # 添加新特征
    data['Remodeled'] = (data['YearBuilt'] != data['YearRemodAdd']) * 1
    data['Age'] = data['YrSold'] - data['YearBuilt'] + 1
    data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']

    # # 数据类型处理
    data[['YrSold','MoSold']] = data[['YrSold','MoSold']].astype(str)
    data['GarageYrBlt'] = data['GarageYrBlt'].astype(str)
    ids = data['Id']
    data = data.drop('Id',axis=1)
    # print('--nan--',np.where(np.isnan(data))) # 对于object类会报错
    print(pd.isnull(data))
    # 偏态数据处理
    numeric_feats = data.drop(['AVG_PRICE','SalePrice'],axis=1).dtypes[(data.dtypes != 'object') & (data.dtypes != 'datetime64[ns]')].index # 获取数值列
    '''skewed_feats = data[numeric_feats].apply(lambda x: skew(x))
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index'''
    std = StandardScaler()
    # data[skewed_feats] = std.fit_transform(data[skewed_feats])
    data[numeric_feats] = std.fit_transform(data[numeric_feats])

    data = pd.get_dummies(data)
    data['Id'] = ids
    return data

if __name__ == "__main__":
    train_data,test_data,data = load_data()
    data = manage_data(data,train_data,test_data)
    test_data = data[data['SalePrice'].isnull()]
    train_data = data[~data['SalePrice'].isnull()]
    print(train_data.shape,test_data.shape)
    test_data = test_data.drop('SalePrice',axis=1)
    test_data.to_csv(PATH + 'step_test.csv',index=False)
    train_data.to_csv(PATH + 'step_train.csv',index=False)


