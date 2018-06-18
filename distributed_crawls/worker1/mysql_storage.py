'''
存储帖子名与帖子url
'''
import pymysql

class MysqlStorage:
    def __init__(self,con=None):
        self.conn = pymysql.connect(host = "192.168.1.5",port = 3306,user = "worker1",passwd = "123456",db = "spider",charset='utf8') if con is None else con
        self.cursor = self.conn.cursor()
        sql = """
                CREATE TABLE IF NOT EXISTS DATA2 (
                    URL VARCHAR(255) NOT NULL PRIMARY KEY,
                    NAME VARCHAR(255) NOT NULL
                )CHARSET=utf8;
                """
        try:
            self.cursor.execute(sql) # 将url设置为主键，便不能重复
        except Exception as e:
            print(e)

    def insert(self,name,url):
        sql = """
                INSERT INTO DATA2(
                    NAME,URL
                ) VALUES (%r,%r) ON DUPLICATE KEY UPDATE NAME= %r;
            """% (name,url,name)
        self.cursor.execute(sql)
        self.conn.commit()
