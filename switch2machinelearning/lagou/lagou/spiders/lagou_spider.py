import scrapy
import json
from urllib.parse import quote
import pandas as pd 
import requests
from lxml import html 

class LagouSpider(scrapy.Spider):
    name = "lagou"
    def __init__(self):
        self.kd = '机器学习'
        self.citys = [
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
            '烟台','兰州','泉州'
        ]
        self.headers = {
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Connection': 'keep-alive',
        'Content-Length': '55',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'Cookie': 'user_trace_token=20170621235533-0f4f9bb1-569a-11e7-b3bc-525400f775ce; LGUID=20170621235533-0f4fa1d1-569a-11e7-b3bc-525400f775ce; _ga=GA1.2.2139591845.1498060522; LG_LOGIN_USER_ID=f2136fa842a914be63b65ddae73c0b3027d684348da01b38; index_location_city=%E5%85%A8%E5%9B%BD; _gid=GA1.2.337438097.1529317916; JSESSIONID=ABAAABAAADEAAFI5F36E8C903B26362AAABD436F70B5AB8; Hm_lvt_4233e74dff0ae5bd0a3d81c6ccf756e6=1527338381,1529317915,1529413880; TG-TRACK-CODE=index_search; LGRID=20180619211213-61c0ebd8-73c2-11e8-a9a6-525400f775ce; Hm_lpvt_4233e74dff0ae5bd0a3d81c6ccf756e6=1529413911; SEARCH_ID=535884639e3246c09ba1f6ebb0c0525a',
        'Host': 'www.lagou.com',
        'Origin': 'https://www.lagou.com',
        'Referer': 'https://www.lagou.com/jobs/list_%s?labelWords=&fromSearch=true&suginput=' % quote(self.kd),
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.87 Safari/537.36',
        'X-Anit-Forge-Code': '0',
        'X-Anit-Forge-Token': 'None',
        'X-Requested-With': 'XMLHttpRequest',
        }

    def start_requests(self):
        for city in self.citys:
            for pn in range(30):
                data = {
                'first':'true','pn':'%d'% pn,'kd':self.kd,
                }
                request_url = 'https://www.lagou.com/jobs/positionAjax.json?px=default&city=%s&needAddtionalResult=false' % city
                yield scrapy.FormRequest(url=request_url,formdata = data,headers = self.headers,callback=self.parse)
    
    def parse(self,response):
        data = json.loads(response.body_as_unicode())
        position_data = data['content']['positionResult']['result']
        for position in position_data:
            yield position