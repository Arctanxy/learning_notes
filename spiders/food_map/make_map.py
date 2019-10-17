import pandas as pd 
import matplotlib.pyplot as plt 
import re
from tqdm import tqdm 

def draw_streets():
    streets = pd.read_csv('street_data.csv')
    streets = streets[streets['COORDINATE'].apply(lambda x:True if x is not None else False)]
    streets = streets[streets['XZQHSZ_DM'].apply(lambda x:True if x in ['"340102"','"340103"','"340104"','"340111"'] else False)] # 筛选合肥四区的街道
    # 将字符串格式的坐标转化为易于处理的列表形式
    def clean(x):
        x = re.sub(r'"|;$','',x)
        coords = x.split(';')
        return [(float(c.split(',')[0]),float(c.split(',')[1])) for c in coords[:-1]]  # 最后会出现一个空字符串
    streets['COORDINATE'] = streets['COORDINATE'].map(clean)
    for s,row in tqdm(streets.iterrows()):
        coords = row['COORDINATE']
        for i in range(len(coords)-1):
            plt.plot([(coords[i][0]),coords[i+1][0]],[coords[i][1],coords[i+1][1]],color = 'r')
    res = pd.read_csv("new_items.csv")
    res = res[res['XZQHSZ_DM'].apply(lambda x: True if x in ["蜀山区","包河区","庐阳区","瑶海区"] else False)]
    '''for c in res['COORDINATE'].values:
        plt.scatter(float(c.split(',')[0]),float(c.split(',')[1]),color= 'g')'''
    res['RATING'] = res['RATING'].fillna(0)
    alphas = {
        10:1,9:1,8:0.8,7:0.8,6:0.5,5:0.5,4:0.3,4:0.3,3:0.15,2:0.15,1:0.15,0:0.05
    }
    for i,row in tqdm(res.iterrows()):
        c = row['COORDINATE']    
        plt.scatter(float(c.split(',')[0]),float(c.split(',')[1]),color= 'g',alpha=alphas[int(row['RATING']*2)])
    plt.show()
    return streets
if __name__ == "__main__":
    draw_streets()

