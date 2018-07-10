'''
Python 拟合任意形式的公式
方法二：双因素n&N
'''
import numpy as np
from sklearn.linear_model import RidgeCV
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


def read_data(floor_range = [8,18]):
    data = pd.read_excel(r'C:\Users\Dl\Desktop\宿州市住宅类可比实例入库版20180529.xlsx')
    data = data[4300:]
    data = data[(data['总楼层']>=floor_range[0])&(data['总楼层']<=floor_range[1])]
    pivot_price = {}
    for name,group in data.groupby('小区名称'):
        pivot_price[name] = group['单价'].mean()
    data['pivot'] = data['小区名称'].apply(lambda x: pivot_price[x])
    data['ratio_price'] = data['单价']/data['pivot']
    # data = data[data['ratio_price'] <= 1.2]
    x = data[['所在楼层','总楼层']]
    y = data['ratio_price']
    pf = PolynomialFeatures(degree=2)
    x = pf.fit_transform(x)
    return x,y

def show_data():
    x,y = read_data()
    clf = RidgeCV(alphas=[1e-5,1e-4,1e-3,1e-2,1e-1,1])
    clf.fit(x,y)
    yp = clf.predict(x)
    plt.scatter(x[:,1], y, alpha=0.4)
    plt.scatter(x[:,1],yp,alpha=0.4)

    plt.show()

if __name__ == "__main__":
    show_data()



