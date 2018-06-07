from celery import Celery
from sqlalchemy import create_engine
import requests
import lxml
import pandas as pd 
import pymysql

# app = Celery('tasks',backend='rpc://',broker='pyamqp://guest@localhost//')

app = Celery('tasks',backend='db+mysql://root:123456@localhost/celery',
                broker='pyamqp://guest@localhost//')

@app.task
def spider():
    url = requests.get('https://www.baidu.com')
    r = requests.get(url)
    df = pd.DataFrame({
        'url':url,'content':r.text
    })
    # cn = create_engine('pymysql://root:123456@localhost/celery')
    # df.to_sql('new',con=cn,if_exists='append')
    return r.text

