import pandas as pd 
import numpy as np 
from scipy.stats import skew
from tqdm import tqdm
PATH = 'H:/learning_notes/project/kaggle/houseprice/data/'

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
        if neighbor.shape[0] <= 5:
            reference_price = neighbor['SalePrice'].mean()
            references.append(reference_price)
            continue
        neighbor['diff_time'] = neighbor['SaleTime'].apply(lambda x:(x-saletime).days) # 计算周边房屋与待评估房屋的销售时间差
        neighbor = neighbor.sort_values(by='diff_time')
        reference_price = neighbor[:5]['SalePrice'].mean()
        references.append(reference_price)
    df['ReferencePrice'] = references
    return df

def manage_nan(df):
    '''
    这一步可以将test和train合并进行处理
    '''
    # 参考https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard

if __name__ == "__main__":
    train_data = pd.read_csv(PATH + 'train.csv')
    test_data = pd.read_csv(PATH + 'test.csv')
    train_data = combine_time(train_data)
    train_data = Add_reference_price(train_data,train_data)
    test_data = combine_time(test_data)
    test_data = Add_reference_price(test_data,train_data)
    print(test_data)
    house = pd.concat([train_data,test_data],axis=0)
