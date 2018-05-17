# 特征工程
# dalalaa
import pandas as pd 
import numpy as np 
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# 读取数据

# PATH = 'H:/learning_notes/project/kaggle/houseprice/data/'
PATH = 'C:/Users/Dl/Documents/GitHub/learning_notes/project/kaggle/houseprice/data/'
def load_data():
    train_data = pd.read_csv(PATH + 'train.csv')
    test_data = pd.read_csv(PATH + 'test.csv')
    data = pd.concat([train_data,test_data],axis=0)
    return train_data,test_data,data

def manage_data(data,train_data,test_data):

    # 等级替换
    data = data.replace({
        'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1
    })
    
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


