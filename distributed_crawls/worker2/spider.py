from mysql_queue import MysqlQueue
import pymysql
import requests
from lxml import html
from urllib.parse import quote
from tqdm import tqdm
import re

class Spider():
    def __init__(self,url=None):
        self.que = MysqlQueue()
        self.path = 'D:/data/'
        self.iterr = 200
        self.domain = 'tieba.baidu.com'
        
    def extract_urls(self):
        '''
        从网页中提取所有网址并导入到数据库中
        '''
        self.request_url()
        r = requests.get('http://' + self.url)
        r.encoding = 'utf-8'
        content = r.content # 使用text时会出现解码错误
        # 将网页下载到本地
        self.download(content)
        tree = html.fromstring(content)
        url_list = tree.xpath('//@href')
        for url in url_list:
            if 'http://' in url or 'https://' in url or '//' in url:
                url = re.sub(r'https:\/\/|http:\/\/|\/\/','',url)
                # 只抓取帖子页面
                if re.search(r'^/p/',url): # 帖子
                    url = self.domain + url
                    self.que.put(quote(url))
                elif re.search(r'^/f\?kw',url): # 帖吧
                    url = self.domain + url
                    self.que.put(quote(url))
                elif 'tieba.baidu.com/' in url:
                    self.que.put(quote(url))
                else:
                    pass

    def run(self):
        '''
        爬虫主体
        '''
        for i in tqdm(range(self.iterr)):
            try:
                self.extract_urls()
            except Exception as e:
                print(e)
            

    def download(self,content):
        filename = self.url.replace('/','_') + '.html'
        with open(self.path + filename,'wb') as f:
            f.write(content)

    def request_url(self):
        '''
        从库中申请新的url
        '''
        self.url = self.que.pop()
    

if __name__ == "__main__":
    spider = Spider()
    spider.run()
