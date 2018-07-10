'''
使用scipy.optimize.leastsq()进行楼层参数的拟合，拟合之后的参数进行归一化之后不影响最终结果
1. 因为一楼和顶楼不符合整体规律，所以本次拟合过程中不包括一楼和顶楼
'''


import pandas as pd
import numpy as np 
from scipy.optimize import leastsq
from tqdm import tqdm
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter('ignore')

def get_data(is_type = '高层住宅'):
    data = pd.read_excel("H:/learning_notes/SUZHOU_CHANGE_EQUATION/可比实例带系数.xlsx")[4300:]
    data = data[data['所在楼层'] != 1]
    # data = data[(data['类型'] == '高层住宅')|(data['类型'] == '中高层住宅')|(data['类型'] == '超高层住宅')]
    data = data[data['类型'] == is_type]
    # data = data[(data['所在楼层'] != 1)&(data['所在楼层'] != data['总楼层'])]
    return data

def evaluation(df,p):
    '''
    评估单个小区价格
    '''
    df['ratio'] = df['所在楼层']/data['总楼层']
    # 凑系数
    df['xishu'] = p # 直接计算所有的楼层修正系数值
    # 凑方程
    # a,b,c = p
    # xishu = a*df['ratio']*df['ratio'] + b * df['ratio'] + c
    # p = xishu.values
    # df['比准价格'] = df['单价']*1e10/(df['xishu']*df['朝向系数']*df['成新系数']*df['结构系数']*df['面积系数'])
    # 凑方程
    # bizhun_price = df['单价']*1e12/(p*df['朝向系数']*df['成新系数']*df['结构系数']*df['面积系数']*df['装修系数'])
    # 凑系数
    bizhun_price = df['单价'] * 1e12 / (df['xishu'] * df['朝向系数'] * df['成新系数'] * df['结构系数'] * df['面积系数'] * df['装修系数'])
    jizhun_price = bizhun_price.mean()
    # 凑方程
    # pinggu_price = (p*df['朝向系数']*df['成新系数']*df['结构系数']*df['面积系数']*df['装修系数']) * jizhun_price/1e12
    # 凑系数
    pinggu_price = (df['xishu']* df['朝向系数'] * df['成新系数'] * df['结构系数'] * df['面积系数'] * df['装修系数']) * jizhun_price / 1e12
    # df['基准价格'] = df['比准价格'].mean()
    # df['评估价格'] = df['基准价格'] * (df['xishu']*df['朝向系数']*df['成新系数']*df['结构系数']*df['面积系数'])/1e10
    # return df['评估价格'].values
    return pinggu_price

def residual(p,y,x):
    return y-evaluation(x,p)

def optimizer(x,y,p0):
    plsq = leastsq(residual,p0,args=(y,x)) # ,ftol=1e-3,xtol=1e-3,maxfev=1000)
    yp = evaluation(x,plsq[0])

    return yp,plsq[0]

def process(data):
    df = pd.DataFrame()
    for name,group in tqdm(data.groupby(['小区名称','类型'])):
        result = optimizer(group,group['单价'],[100 for i in range(group.shape[0])])
        # result = optimizer(group, group['单价'], [20,30,40])
        group['评估价格'] = result[0]
        # 凑系数
        group['xishu'] = result[1]
        # 凑方程
        # group['xishu'] = result[1][0] * group['ratio'] * group['ratio'] + result[1][1] * group['ratio'] + result[1][2]
        # group['equation'] = [result[1] for i in range(group.shape[0])]
        df = pd.concat([df,group],axis=0)
    return df

if __name__ == "__main__":
    IS_TYPE =['中高层住宅','高层住宅','超高层住宅']
    for i_type in IS_TYPE:
        print("正在处理%s" % i_type)
        data = get_data(is_type=i_type)
        df = process(data)
        df.to_excel("H:/learning_notes/SUZHOU_CHANGE_EQUATION/%s系数计算结果.xlsx" % i_type,index=False)


    '''data = get_data()
    data = data[data['小区名称'] == '两淮融景苑 20.21.24.25#(33)']
    x = data
    y = data['单价']
    p = [100 for i in range(data.shape[0])]
    result = optimizer(x,y,p)
    print(result,y)'''