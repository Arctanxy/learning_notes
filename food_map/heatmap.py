import pandas as pd 
import matplotlib.pyplot as plt 
from tqdm import tqdm 
from point_in_polygon import pointInPolygon
from fill import fill
import re 

df = pd.read_csv('C:\\Users\\Dl\\Documents\\GitHub\\learning_notes\\food_map\\new_items.csv')
df['RATING'] = df['RATING'].fillna(df['RATING'].mean())
df['COORDINATE'] = df['COORDINATE'].apply(lambda x:eval('('+str(x) + ')'))
streets = pd.read_csv('C:\\Users\\Dl\\Documents\\GitHub\\learning_notes\\food_map\\street_data.csv')
streets = streets[streets['COORDINATE'].apply(lambda x:True if x is not None else False)]
streets = streets[streets['XZQHSZ_DM'].apply(lambda x:True if x in ['"340102"','"340103"','"340104"','"340111"'] else False)]
# 将字符串格式的坐标转化为易于处理的列表形式
def clean(x):
    x = re.sub(r'"|;$','',x)
    coords = x.split(';')
    return [(float(c.split(',')[0]),float(c.split(',')[1])) for c in coords[:-1]]  # 最后会出现一个空字符串
streets['COORDINATE'] = streets['COORDINATE'].map(clean)


# 匹配美食所属街道
belong_street = []
for i,row1 in tqdm(df.iterrows()):
    b_street = ""
    for j,row2 in streets.iterrows():
        if pointInPolygon(row1['COORDINATE'],row2['COORDINATE']):
            b_street = row2['QH_CODE']
            break
    belong_street.append(b_street)
        
df['QH_CODE'] = belong_street


# 计算每个街道的平均评分
avg_ratings = []
for h,row in tqdm(streets.iterrows()):
    code = row['QH_CODE']
    avg_ratings.append(round(df[df['QH_CODE']== code]['RATING'].mean(),1))
streets['AVG_RATING'] = avg_ratings

print(streets['AVG_RATING'].value_counts())

# 填充
fill(streets)

# 绘制地图边界
for s,row in tqdm(streets.iterrows()):
        coords = row['COORDINATE']
        for i in range(len(coords)-1):
            plt.plot([(coords[i][0]),coords[i+1][0]],[coords[i][1],coords[i+1][1]],color = 'black')
plt.show()
