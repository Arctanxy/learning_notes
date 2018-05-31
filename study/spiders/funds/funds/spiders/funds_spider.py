import scrapy
import json

class FundsSpider(scrapy.Spider):
    name = 'funds'
    allowed_domains = ['fund.eastmoney.com']


    def start_requests(self):
        url = 'https://fundapi.eastmoney.com/fundtradenew.aspx?ft=pg&sc=1n&st=desc&pi=1&pn=3000&cp=&ct=&cd=&ms=&fr=&plevel=&fst=&ftype=&fr1=&fl=0&isab='
        yield scrapy.Request(url,callback=self.parse_fund)

    def parse_fund(self,response):
        datas = response.body.decode('UTF-8')
        yield datas