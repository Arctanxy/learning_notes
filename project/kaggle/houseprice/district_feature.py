import pandas as pd 
import xgboost as xgb
from sklearn.linear_model import LassoCV
from features import PATH
import seaborn as sns
import matplotlib.pyplot as plt 

'''
将房屋因素分为两类：

1. 房屋外部因素，主要是小区因素和地理因素
2. 房屋内部因素，除了外部都是内部
'''
# 有多个类别的因素可以根据价格分五级




if __name__ == "__main__":
    test_data = pd.read_csv(PATH + 'test.csv')
    train_data = pd.read_csv(PATH + 'train.csv')

    sns.distplot(test_data['YrSold'])

    plt.show()