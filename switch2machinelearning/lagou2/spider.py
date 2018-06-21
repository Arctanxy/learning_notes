import pandas as pd 
import json
import requests
from lxml import html
from urllib.parse import quote
import time 
import random



def crawler(keyword='机器学习',city='北京',pn = 1):
    url = 'https://www.lagou.com/jobs/positionAjax.json?px=default&city=%s&needAddtionalResult=false' % city
    headers = {
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Connection': 'keep-alive',
        'Content-Length': '55',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'Cookie': 'user_trace_token=20170621235533-0f4f9bb1-569a-11e7-b3bc-525400f775ce; LGUID=20170621235533-0f4fa1d1-569a-11e7-b3bc-525400f775ce; _ga=GA1.2.2139591845.1498060522; LG_LOGIN_USER_ID=f2136fa842a914be63b65ddae73c0b3027d684348da01b38; index_location_city=%E5%85%A8%E5%9B%BD; _gid=GA1.2.337438097.1529317916; JSESSIONID=ABAAABAAAGGABCB25D75E024F591195DF2E8759F3614172; LGSID=20180620204317-8190680a-7487-11e8-9727-5254005c3644; PRE_UTM=; PRE_HOST=www.google.com; PRE_SITE=https%3A%2F%2Fwww.google.com%2F; PRE_LAND=https%3A%2F%2Fwww.lagou.com%2F; _gat=1; Hm_lvt_4233e74dff0ae5bd0a3d81c6ccf756e6=1529413880,1529498575,1529499507,1529499516; TG-TRACK-CODE=index_search; Hm_lpvt_4233e74dff0ae5bd0a3d81c6ccf756e6=1529499932; LGRID=20180620210555-ab130d07-748a-11e8-9727-5254005c3644; SEARCH_ID=a7c11bd64e014c8cad8ac5fa2e7deea7',
        'Host': 'www.lagou.com',
        'Origin': 'https://www.lagou.com',
        'Referer': 'https://www.lagou.com/jobs/list_%s?px=default&city=%s' % (quote(keyword),quote(city)),
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.87 Safari/537.36',
        'X-Anit-Forge-Code': '0',
        'X-Anit-Forge-Token': 'None',
        'X-Requested-With': 'XMLHttpRequest',
    }
    data = {
        'first':'true','pn':'%d' % pn,'kd':keyword
    }
    r = requests.post(url,data=data,headers = headers)
    position = json.loads(r.text)
    result = position['content']['positionResult']['result']
    df = pd.DataFrame()
    for r in result:
        for k,v in r.items():
            r[k] = [str(v)]
        data = pd.DataFrame(r)
        df = pd.concat([df,data],axis=0)
    return df

def run():
    citys = [
            '北京','上海','深圳','广州',
            '杭州','成都','南京','武汉',
            '西安','厦门','长沙','苏州',
            '重庆','郑州','青岛','合肥',
            '福州','济南','大连','珠海',
            '无锡','佛山','东莞','宁波',
            '常州','沈阳','石家庄','昆明',
            '南昌','南宁','哈尔滨','海口',            
            '中山','惠州','贵阳','长春',                                  
            '太原','嘉兴','泰安','昆山',
            '烟台','兰州',
        ]
    df = pd.DataFrame()
    for city in citys:
        for pn in range(30):
            time.sleep(1+1.5*random.random())
            data = crawler(city=city,pn=(pn+1))
            df = pd.concat([data,df],axis=0)
            print(df.shape)
            if data.shape[0] < 15:
                break
            
    return df

if __name__ == "__main__":
    df = run()
    df.to_excel("H:/learning_notes/switch2machinelearning/lagou2/machine_learning_data.xlsx",index=False)