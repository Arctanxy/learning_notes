import scrapy
import json
import re
import urllib.parse

class InfluentSpider(scrapy.Spider):
    
    name = "influent"

    def gen_url(self):
        SITES = {
            'BANK':'银行','HOSPITAL':'医院','SUPERMARKET':'超市','SHOPPING':'商场',
            'SCENIC':'景区','PARK':'公园','SCHOOL':'学校','KINDERGARDEN':'幼儿园',
            'FACTORY':'工厂','HOTEL':'酒店','RESTAURANT':'美食'
        }
        GRID_DICT = {
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
        CITY = 'HANSHAN'
        length = 4
        grid = GRID_DICT[CITY]
        lat = ['%r' % (grid[0][0] + (grid[2][0] - grid[0][0]) * i/length ) for i in range(length -1)]
        lng = ['%r' % (grid[2][1] + (grid[0][1] - grid[2][1]) * i/length ) for i in range(length -1)]
        points = []
        for a in lat:
            for n in lng:
                points.append(n+','+a)
        urls = []
        for k,v in SITES.items():
            for p in points:
                urls.append(
                    'http://api.map.baidu.com/place/v2/search?query=%s&location=%s' \
                    '&radius=5000&output=json&ak=f8QB8NM61oZ2j1FuotFa4z9PB9lUqRwE' % (v,p)
                )                
        return urls


    def start_requests(self):
        urls = self.gen_url()
        for url in urls:
            yield scrapy.Request(url=url,callback=self.parse)
    
    def parse(self,response):
        is_type = re.findall(r'query=(.*?)&',response.url)[0]
        is_type = urllib.parse.unquote(is_type)
        information = json.loads(response.body_as_unicode())['results']
        if information != []:
            for infor in information:
                yield {
                    'is_type':is_type,
                    'name':infor['name'],
                    'coord':str(infor['location']['lng'])+','+str(infor['location']['lat']),
                    'district':infor['area'],
                    'address':infor['address']
                }
        else:
            pass
