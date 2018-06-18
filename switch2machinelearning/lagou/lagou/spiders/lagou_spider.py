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
        pass

    def start_requests(self):
        
        for city in self.citys:
            data = {
            'first':'true','pn':'1','kd':'机器学习','Referer':'Referer: https://www.lagou.com/jobs/list_%s?px=default&city=%s'%(quote('机器学习'),quote(city))
            }
            request_url = 'https://www.lagou.com/jobs/positionAjax.json?px=default&city=%s&needAddtionalResult=false' % city
            yield scrapy.FormRequest(url=request_url,formdata = data,callback=self.parse)
    
    def parse(self,response):
        print(json.loads(response.body_as_unicode()))