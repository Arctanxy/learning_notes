import pymysql
from sqlalchemy import create_engine

'''
管理url队列
'''

class MysqlQueue():
    def __init__(self,conn = None):
        self.conn = pymysql.connect("localhost","root","123456","celery") if conn is None else conn
        self.cursor = self.conn.cursor()
        sql = """
                CREATE TABLE DATA1 (
                    URL TEXT,
                    USED INT NOT NULL
                )CHARSET=gbk;
                """
        try:
            self.cursor.execute(sql) # 将url设置为主键，便不能重复
        except Exception as e:
            print(e)
        
    def put(self,url):
        # 主键重复可能会报错
        try:
            self.cursor.execute("""
                INSERT INTO DATA1(
                    URL,USED
                ) VALUES (%r,0)
                """ % url)
            self.conn.commit()
        except Exception as e:
            print(url)
            print(e)
        
        

    def pop(self):
        # 提取之后将used字段置为1
        self.cursor.execute("""
                SELECT TOP 1 URL FROM DATA1
                """)
        url = self.cursor.fetchone()
        self.cursor.execute("""
                UPDATE DATA1 SET USED = true WHERE URL=%s
                """ % url)
        return url
    
