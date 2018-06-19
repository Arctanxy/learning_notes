import pandas as pd 
import json
import requests
from lxml import html
from urllib.parse import quote

def crawler(keyword='机器学习',city='全国',pn = 1):

    url = 'https://www.lagou.com/jobs/positionAjax.json?needAddtionalResult=false'

    headers = {
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Connection': 'keep-alive',
        'Content-Length': '55',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'Cookie': 'user_trace_token=20170621235533-0f4f9bb1-569a-11e7-b3bc-525400f775ce; LGUID=20170621235533-0f4fa1d1-569a-11e7-b3bc-525400f775ce; _ga=GA1.2.2139591845.1498060522; LG_LOGIN_USER_ID=f2136fa842a914be63b65ddae73c0b3027d684348da01b38; index_location_city=%E5%85%A8%E5%9B%BD; _gid=GA1.2.337438097.1529317916; JSESSIONID=ABAAABAAADEAAFI5F36E8C903B26362AAABD436F70B5AB8; Hm_lvt_4233e74dff0ae5bd0a3d81c6ccf756e6=1527338381,1529317915,1529413880; TG-TRACK-CODE=index_search; LGRID=20180619211213-61c0ebd8-73c2-11e8-a9a6-525400f775ce; Hm_lpvt_4233e74dff0ae5bd0a3d81c6ccf756e6=1529413911; SEARCH_ID=535884639e3246c09ba1f6ebb0c0525a',
        'Host': 'www.lagou.com',
        'Origin': 'https://www.lagou.com',
        'Referer': 'https://www.lagou.com/jobs/list_%s?labelWords=&fromSearch=true&suginput=' % quote(keyword),
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