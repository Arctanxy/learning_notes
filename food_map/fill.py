import pandas as pd 
import matplotlib.pyplot as plt 
from tqdm import tqdm

def cross_point(polygon,num):
    '''polygon是多边形点列表，num是水平线的y坐标
    return 交点坐标
    '''
    ps = []
    for i in range(len(polygon)-1):
        p1 = polygon[i]
        p2 = polygon[i+1]
        if max(p1[1],p2[1]) < num or min(p1[1],p2[1]) > num:
            continue
        #如果p1p2不是水平线
        if p1[1] != p2[1]:
            x = (num-p1[1])*(p2[0]-p1[0])/(p2[1]-p1[1])+p1[0]
        #如果p1p2是水平线，水平线不求交点
        if p1[1] == p2[1]:
            x = None
        #左闭右开，但是要排除竖直线的情况，因为竖直线时，x==p1[0] and x==p2[0]
        #if x == p2[0] and x != p1[0]:
        if x== p2[0] and num == p2[1]:
            x = None
        #判断是不是上下顶点
        if i == 0:
            p0 = polygon[len(polygon)-2]
        else:
            p0 = polygon[i-1]
        #如果是上下顶点
        if x == p1[0]:
            if (p0[1]>p1[1] and p2[1]>p1[1]) or (p0[1]<p1[1] and p2[1]<p1[1]):
                x = None
        if x != None:
            if x==2.0:
                print(p1[0],p1[1],p2[0],p2[1])
            ps.append((x,num))
    return ps

def fill(df,col = 'AVG_RATING'):
    '''填充'''
    nums = [31.5000+i/3000 for i in range(3000)]
    #df['COORDINATE'] = df['COORDINATE'].map(seg_to_point)
    #for poly in tqdm(list(df['COORDINATE'].map(seg_to_point).values)):
    #红黄绿篮紫
    for i,row in tqdm(df.iterrows()):
        poly = row['COORDINATE']
        alpha = 0.5
        if row['AVG_RATING']>= 4.5:
            color = 'red'
        elif row['AVG_RATING']> 4.3 and row['AVG_RATING']< 4.5:
            # color = 'orangered'
            color = 'tomato'
        elif row['AVG_RATING']> 4.2 and row['AVG_RATING']<=4.3:
            # color = 'tomato'
            # color = 'yellow'
            color = 'orange'
        elif row['AVG_RATING']> 4.1 and row['AVG_RATING']<=4.2:
            # color = 'salmon'
            # color = 'green'
            color = 'yellow'
        elif row['AVG_RATING']> 3.9 and row['AVG_RATING']<=4.1:
            # color = 'lightsalmon'
            color = 'greenyellow'
        else:
            # color = 'mistyrose'
            color = 'lightblue'
        for num in nums:
            cps = cross_point(poly,num)
            for i in range(int(len(cps)/2)):
                line = (cps[2*i],cps[2*i + 1])
                plt.plot([line[0][0],line[1][0]],[line[0][1],line[1][1]],color=color,linewidth=3,alpha=alpha)

if __name__ == "__main__":
    pass