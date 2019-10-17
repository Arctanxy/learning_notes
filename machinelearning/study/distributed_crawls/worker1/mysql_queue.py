import pymysql

'''
管理url队列
'''

class MysqlQueue():
    def __init__(self,conn = None):
        self.start_url = 'tieba.baidu.com/index.html'
        self.conn = pymysql.connect(host = "192.168.1.5",port = 3306,user = "worker1",passwd = "123456",db = "spider",charset='utf8') if conn is None else conn
        self.cursor = self.conn.cursor()
        sql = """
                CREATE TABLE IF NOT EXISTS DATA1 (
                    URL VARCHAR(255) NOT NULL PRIMARY KEY,
                    USED INT NOT NULL
                )CHARSET=utf8;
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
                ) VALUES (%r,0) ON DUPLICATE KEY UPDATE URL=%r
                """ % (url,url))
            self.conn.commit()
        except Exception as e:
            print(e)
        
        

    def pop(self):
        # 提取之后将used字段置为1
        self.cursor.execute("""
                SELECT URL FROM DATA1 WHERE USED=0 LIMIT 1
                """)
        url = self.cursor.fetchone()
        if url is None:
            url = self.start_url
        else:
            url = url[0]
        self.cursor.execute("""
                UPDATE DATA1 SET USED=1 WHERE URL=%r
                """ % url)
        self.conn.commit()
        return url
    
