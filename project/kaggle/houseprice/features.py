import pandas as pd 
import numpy as np 
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
PATH = 'H:/learning_notes/project/kaggle/houseprice/data/' # For Windows1
# PATH = '~/Documents/Github/learning_notes/project/kaggle/houseprice/data/' # For Windows2
# PATH = '~/houseprice/data/' # For Linux

def combine_time(df):
    '''
    将月份与年份合并
    '''
    data = pd.DataFrame({
        'year':df['YrSold'],
        'month':df['MoSold'],
        'day':[1 for i in range(df.shape[0])]
    })
    df_time = pd.to_datetime(data,format='%Y%m') # 转化为时间格式
    df['SaleTime'] = df_time
    return df

def Add_reference_price(df,train_data):
    '''
    df可以是train_data和test_data
    '''
    references = []
    print('--正在计算参考价格--')
    for i,row in tqdm(df.iterrows()):
        saletime = row['SaleTime'] # 销售时间
        region = row['Neighborhood'] # 房屋所处区域
        neighbor = train_data[train_data['Neighborhood'] == region]
        if neighbor.shape[0] <= 3:
            reference_price = neighbor['SalePrice'].mean()
            references.append(reference_price)
            continue
        neighbor['diff_time'] = neighbor['SaleTime'].apply(lambda x:(x-saletime).days) # 计算周边房屋与待评估房屋的销售时间差
        neighbor = neighbor.sort_values(by='diff_time')
        reference_price = neighbor[:3]['SalePrice'].mean()
        references.append(reference_price)
    df['ReferencePrice'] = references
    return df

def manage(df):
    '''
    这一步可以将test和train合并进行处理
    '''
    # 参考https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
    
    

    # np.nan 代表 None
    f_None = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageType',
                'GarageFinish','GarageQual','GarageCond','BsmtQual','BsmtCond','BsmtExposure',
                'BsmtFinType1','BsmtFinType2','MasVnrType','MSSubClass']

    for f1 in f_None:
        df[f1] = df[f1].fillna('None')
    # np.nan 代表 0
    f_0 = ['GarageYrBlt','GarageArea','GarageCars','BsmtFinSF1', 'BsmtFinSF2', 
                'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath','MasVnrArea']

    for f2 in f_0:
        df[f2] = df[f2].fillna(0)
    # 连续型数值变量以中位数填充
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

    # 类目型变量或离散型数值变量使用众数填充
    f_mode = ['MSZoning','Electrical','KitchenQual','Exterior1st','Exterior2nd',
                'SaleType']
    for f3 in f_mode:
        df[f3] = df[f3].fillna(df[f3].mode()[0]) # 众数可能不止一个

    # data_description 中说明 Functional特征的np.nan代表"Typ"
    df['Functional'] = df['Functional'].fillna('Typ')

    # Utilities特征中除了两个np.nan 和一个"NoSeWa"外，全是"AllPub"，这种特征可以直接删去
    # 删除无用列

    df = df.drop(['Street','Utilities','Condition2','RoofMatl','Heating'],axis=1)

    # 再添加其他特征
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['Age'] = 2012 - df['YearBuilt']
    df['Remodeled'] = (df['YearBuilt'] != df['YearRemodAdd']) * 1
    df['RecentRemodel'] = (df['YearBuilt'] == df['YearRemodAdd']) * 1
    df['TimeSinceSold'] = 2012 - df['YrSold']
    area_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
             'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 
             'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'LowQualFinSF', 'PoolArea' ]
    df["TotalArea"] = df[area_cols].sum(axis=1)
    # 参考 https://www.kaggle.com/thevachar/house-price-regression-and-feature-engineering

   

    # 将类目型特征转化为标准化的离散数值变量
    cols = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold']
    print(cols)
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(df[c].values))
        df[c] = lbl.transform(list(df[c].values))
    
    # 将一些数值型的类目特征转化为字符串格式
    f_ca = ['MSSubClass','OverallCond','YrSold','MoSold']
    for f4 in f_ca:
        df[f4] = df[f4].astype(str)
    
    # 处理偏态分布数据
    numeric_feats = df.dtypes[(df.dtypes != 'object') & (df.dtypes != 'datetime64[ns]')].index # 获取数值列
    numeric_feats = numeric_feats.drop(['ReferencePrice','Id'])
    df['ReferencePrice'] = np.log(df['ReferencePrice']) # 参考价格采用与最终售卖价格同样的处理策略
        
    print(numeric_feats)
    skewed_feats = df[numeric_feats].apply(lambda x: skew(x))
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index
    std = StandardScaler()
    df[skewed_feats] = std.fit_transform(df[skewed_feats]) # 将数据转为正态分布，保证了train和test符合同一分布
    # df[skewed_feats] = np.log(df[skewed_feats]+1 ) # +1是为了防止出现inf
    print(df.info())
    df = pd.get_dummies(df)
    print(df.shape)

    df = df.drop('MSZoning_C (all)',axis=1)
    return df

def manage_outlier(df):
    '''
    删除异常值，只针对train_data
    '''
    df = df.drop(df[(df['GrLivArea'] > 4000) & (df['SalePrice']<300000)].index)
    return df



if __name__ == "__main__":
    train_data = pd.read_csv(PATH + 'train.csv')
    train_data = combine_time(train_data)
    train_data['SalePrice'] = np.log(train_data['SalePrice'])
    train_data = manage_outlier(train_data)
    print('train shape',train_data.shape) # 去除outlier之后的train大小，方便后面分割数据用
    test_data = pd.read_csv(PATH + 'test.csv')
    test_data = combine_time(test_data)
    house = pd.concat([train_data,test_data],axis=0)
    house = Add_reference_price(house,train_data)
    house = manage(house)
    test = house[1456:]
    train = house[:1456]
    test.to_csv(PATH + 'new_test.csv',index=False)
    train.to_csv(PATH + 'new_train.csv',index=False)
    print('train_null' ,train.SalePrice.isnull().sum())
