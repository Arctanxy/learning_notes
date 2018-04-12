import scrapy
import json

from scrapy.http import Request
from funds.items import FundsItem

class FundsSpider(scrapy.Spider):
    name = 'fundsList'#爬虫名称，用于运行爬虫
    allowed_domains = ['fund.eastmoney.com'] #允许访问的域

    def start_requests(self):
        url = 'http://fund.eastmoney.com/fund.html#os_0;isall_0;ft_;pt_1'
        #callback函数用于处理Request的返回值，即response
        request = scrapy.Request(url,callback=self.parse_funds_list)
        return request

    def parse_funds_list(self,response):
        print(response)