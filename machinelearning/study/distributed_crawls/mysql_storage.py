'''
存储网页数据,已弃用
'''

import pymysql 
from mysql_queue import MysqlQueue

class MysqlStorage():
    def __init__(self,conn = None):
        self.conn = pymysql.connect("localhost","root","123456","celery") if conn is None else conn
        self.cursor = self.conn.cursor()
        sql = """
                CREATE TABLE IF NOT EXISTS DATA2 (
                    URL VARCHAR(255) NOT NULL PRIMARY KEY,
                    HTML LONGTEXT 
                )CHARSET=utf8;
                """
        try:
            self.cursor.execute(sql) # 将url设置为主键，便不能重复
        except Exception as e:
            print(e)
    
    def put(self,url,html):
        # 主键重复可能会报错
        try:
            self.cursor.execute("""
                INSERT INTO DATA2(
                    URL,HTML
                ) VALUES (%r,%r)
                """ % (url,html))
            self.conn.commit()
        except Exception as e:
            print(url,e)
