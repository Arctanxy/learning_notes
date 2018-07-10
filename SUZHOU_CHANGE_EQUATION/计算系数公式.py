import pandas as pd
import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

types = ['中高层住宅','高层住宅','超高层住宅']
floor = ['一楼','中间层','顶楼']

def get_data(is_type = types[0],is_floor = floor[0]):
    data = pd.read_excel("D:/SUZHOU_CHANGE_EQUATION/系数计算结果.xlsx")
    data = data[data['类型'] == is_type]
    if floor == '中间层':
        data = data[(data['所在楼层']>1)&(data['所在楼层']!= data['总楼层'])]
    return data

def get_equation(df,p):
    a,b,c = p
    ratio = df['所在楼层']/df['总楼层']
    return a*pow(ratio,2) + b*ratio + c

def residual(p,y,x):
    return y - get_equation(x,p)

def optimizer(x,y,p0):
    plsq = leastsq(residual,p0,args=(y,x))
    yp = get_equation(x,plsq[0])

    return yp,plsq[0]

if __name__ == "__main__":
    df = get_data()
    x = df['所在楼层']/df['总楼层']
    y = df['xishu']