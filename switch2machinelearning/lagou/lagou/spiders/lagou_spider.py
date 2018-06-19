import scrapy
import json
from urllib.parse import quote
import pandas as pd 

class LagouSpider(scrapy.Spider):
    name = "lagou"
    def __init__(self):
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
            'Accept-Encoding': 'gzip,deflate,br',
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'Connection': 'keep-alive',
            'Content-Length': '55',
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'Host': 'www.lagou.com',
            'Origin': 'https://www.lagou.com',
            'X-Anit-Forge-Code': '0',
            'X-Anit-Forge-Token': 'None',
            'X-Requested-With': 'XMLHttpRequest',
        }

    def start_requests(self):
        
        for city in self.citys:
            for pn in range(30):
                data = {
                'first':'true','pn':'%d'% pn,'kd':'机器学习','Referer':'Referer: https://www.lagou.com/jobs/list_%s?px=default&city=%s'%(quote('机器学习'),quote(city))
                }
                request_url = 'https://www.lagou.com/jobs/positionAjax.json?px=default&city=%s&needAddtionalResult=false' % city
                yield scrapy.FormRequest(url=request_url,formdata = data,headers = self.headers,callback=self.parse)
    
    def parse(self,response):
        data = json.loads(response.body_as_unicode())
        position_data = data['content']['positionResult']['result']
        for position in position_data:
            """yield {
                'company_Id':position['company_Id'],
                companyFullName:position['companyFullName'],
                companyLabelList:position['companyLabelList'],
                companySize:position['companySize'],
                position_Id:position['position_Id'],
                education:position['education'],
                financeStage:position['financeStage'],
                firstType:position['firstType'],
                hitags:position['hitags'],
                industryField:position['industryField'],
                city:position['city'],
                
            }"""
            yield position