import pandas as pd 
import requests 
from lxml import html
from urllib.parse import quote
from tqdm import tqdm
import random
import time



def get_info(df,id_col='positionId'):
    description = []
    for i,row in tqdm(df.iterrows()):
        city = row['city']
        position_id = row[id_col]
        headers = {
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'Connection': 'keep-alive',
            'Content-type': 'application/json;charset=utf-8',
            'Cookie': 'user_trace_token=20170621235533-0f4f9bb1-569a-11e7-b3bc-525400f775ce; LGUID=20170621235533-0f4fa1d1-569a-11e7-b3bc-525400f775ce; _ga=GA1.2.2139591845.1498060522; LG_LOGIN_USER_ID=f2136fa842a914be63b65ddae73c0b3027d684348da01b38; index_location_city=%s; _gid=GA1.2.337438097.1529317916; JSESSIONID=ABAAABAAAGGABCB25D75E024F591195DF2E8759F3614172; LGSID=20180620204317-8190680a-7487-11e8-9727-5254005c3644; Hm_lvt_4233e74dff0ae5bd0a3d81c6ccf756e6=1529413880,1529498575,1529499507,1529499516; SEARCH_ID=d11ef7ee59d74437a8a495088d3ba52c; TG-TRACK-CODE=search_code; _gat=1; Hm_lpvt_4233e74dff0ae5bd0a3d81c6ccf756e6=1529503964; LGRID=20180620221307-0e3f8bfc-7494-11e8-ab44-525400f775ce' % quote(city),
            'Host': 'www.lagou.com',
            'Referer': 'https://www.lagou.com/jobs/%d.html' % position_id,
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.87 Safari/537.36',
        }
        time.sleep(2*random.random())
        r = requests.get('https://www.lagou.com/jobs/%d.html' % row[id_col],headers=headers)
        tree = html.fromstring(r.text)
        texts = tree.xpath('//*[@id="job_detail"]/dd[2]/div//p/text()')
        description.append(texts)
    df['DESCRIPTION'] = description
    return df
def run():
    df = pd.read_excel("machine_learning_data.xlsx")
    df = get_info(df)
    df.to_excel("H:/learning_notes/switch2machinelearning/lagou2/machine_learning_data_2.xlsx",index=False)

if __name__ == "__main__":
    run()