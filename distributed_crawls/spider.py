from mysql_queue import MysqlQueue
import requests
from lxml import html
from tqdm import tqdm

que = MysqlQueue()

start_url = 'https://tieba.baidu.com/index.html'

que.put(start_url)

r = requests.get(start_url)
r.encoding = 'gbk'
content = r.content # 使用text时会出现解码错误

tree = html.fromstring(content)

url_list = tree.xpath('//@href')

for url in tqdm(url_list):
    que.put(url)