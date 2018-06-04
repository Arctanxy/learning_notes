import scrapy
import json
import re
import urllib.parse
import pandas as pd 

class InfluentSpider(scrapy.Spider):
    
    name = "influent"
    def __init__(self):
        self.SITES = {
            '银行':'BANK','医院':'HOSPITAL','超市':'SUPERMARKET','商场':'SHOPPING',
            '景区':'SCENIC','公园':'PARK','学校':'SCHOOL','幼儿园':'KINDERGARDEN',
            '工厂':'FACTORY','酒店':'HOTEL','美食':'RESTAURANT'
        }
        # xzqhsz_dm = pd.read_excel('D:/Modeling/ANHUI/xzqhsz_dm.xlsx')
        xzqhsz_dm = pd.read_excel('H:/learning_notes/study/scrapy_project/xzqhsz_dm.xlsx')
        self.xzqhsz_dm = dict(zip(xzqhsz_dm['XZQH'].values,xzqhsz_dm['XZQHSZ_DM'].values))
        
    def gen_url(self):
        
        GRID_DICT = {
            # 合肥
            # 左上角：吴家圩 116.845151,32.004885
            # 右下角：合肥吴家花园 117.475669,31.65874
            'HEFEI':[
                [116.845151,32.004885],
                [116.845151,31.65874],
                [117.475669,31.65874],
                [117.475669,32.004885]
            ],

            # 含山
            # 左上角：祝塘  117.87344,31.914224
            # 右下角：刘圩  118.219395,31.397105
            'HANSHAN':[
                [117.87344,31.914224],
                [117.87344,31.397105],
                [118.219395,31.397105],
                [118.219395,31.914224]
            ],
            # 宁国
            # 左上角：大谈村 118.619455,30.776699
            # 右下角：张家坞 119.43526,30.293009
            'NINGGUO':[
                [118.619455,30.776699],
                [118.619455,30.293009],
                [119.43526,30.293009],
                [119.43526,30.776699]
            ]
        }
        CITY = 'HEFEI'
        length = 80
        grid = GRID_DICT[CITY]
        lat = ['%r' % (grid[0][0] + (grid[2][0] - grid[0][0]) * i/length ) for i in range(length -1)]
        lng = ['%r' % (grid[2][1] + (grid[0][1] - grid[2][1]) * i/length ) for i in range(length -1)]
        points = []
        for a in lat:
            for n in lng:
                points.append(n+','+a)
        urls = []
        for k,v in self.SITES.items():
            for p in points:
                urls.append(
                    'http://api.map.baidu.com/place/v2/search?query=%s&location=%s' \
                    '&radius=1000&output=json&ak=f8QB8NM61oZ2j1FuotFa4z9PB9lUqRwE' % (k,p)
                )                
        return urls


    def start_requests(self):
        urls = self.gen_url()
        for url in urls:
            yield scrapy.Request(url=url,callback=self.parse)
    
    def parse(self,response):
        is_type = re.findall(r'query=(.*?)&',response.url)[0]
        is_type = urllib.parse.unquote(is_type)
        is_type = self.SITES[is_type]
        information = json.loads(response.body_as_unicode())['results']
        if information != []:
            for infor in information:
                yield {
                    'is_type':is_type,
                    'name':infor['name'],
                    'coord':str(infor['location']['lng'])+','+str(infor['location']['lat']),
                    'district':self.xzqhsz_dm[infor['area']],
                    'address':infor['address']
                }
        else:
            pass
