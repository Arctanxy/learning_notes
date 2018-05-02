#直接使用pandas中的quantile方法编写离散化函数
#将所有数据分为五个级别，分别使用0.2,0.4,0.6,0.8分位数
def discretize(series):
    a2 = series.quantile(0.2)
    a4 = series.quantile(0.4)
    a6 = series.quantile(0.6)
    a8 = series.quantile(0.8)
    for i in range(len(series)):
        if series[i]>=0 and series[i] < a2:
            series[i] = 1
        elif series[i]>=a2 and series[i] < a4:
            series[i] =2
        elif series[i]>=a4 and series[i] < a6:
            series[i] =3
        elif series[i]>=a6 and series[i] < a8:
            series[i] =4
        else:
            series[i] = 5
    return series

import pandas as pd

s = pd.Series([1,2,3,4,5,6,7,8,9,10])

s = discretize(s)

print(s)

