'''
Python拟合任意形式的函数
http://python.jobbole.com/87015/
方法一：单因素（n/N）
'''

from scipy.optimize import leastsq
import random
import matplotlib.pyplot as plt
import pandas as pd

def function(x,p):
    a,b,c,d,e,f,g,h,i = p
    return a*x**3 + b*x**2 + c*x + d + e/(f*x**3 + g*x**2 + h*x + i)

def residuals(p,y,x):
    return y-function(x,p)

# 设置初始参数
def init_theta():
    return [random.random() for i in range(9)]

def show_data(x,y):
    plsq = leastsq(residuals,init_theta(),args=(y,x))
    print(plsq,plsq[0])
    yp = function(x,plsq[0])
    plt.scatter(x, y, color='g', alpha=0.5)
    plt.scatter(x,yp,color='r',alpha=0.5)
    plt.show()

def read_data(floor_range = [8,18]):
    data = pd.read_excel(r'C:\Users\Dl\Desktop\宿州市住宅类可比实例入库版20180529.xlsx')
    data = data[4300:]
    data = data[(data['总楼层']>=floor_range[0])&(data['总楼层']<=floor_range[1])]
    data['ratio'] = data['所在楼层']/data['总楼层']
    pivot_price = {}
    for name,group in data.groupby('小区名称'):
        pivot_price[name] = group['单价'].mean()
    data['pivot'] = data['小区名称'].apply(lambda x: pivot_price[x])
    data['ratio_price'] = data['单价']/data['pivot']
    # data = data[data['ratio_price'] <= 1.2]
    x = data['ratio']
    y = data['ratio_price']
    return x,y

if __name__ == "__main__":
    x,y = read_data(floor_range=[34,45])
    show_data(x,y)



