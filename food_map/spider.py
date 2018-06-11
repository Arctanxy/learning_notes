'''
制作合肥美食地图
'''

import requests
import json
import pymysql
import time
from tqdm import tqdm

GRID = {
    '合肥':[
        [116.884841,31.973278],
        [116.884841,31.589222],
        [117.473626,31.589222],
        [117.473626,31.973278]
    ]
}

class MysqlQueue:
    def __init__(self,points,db_name="point_queue", table_name="points",conn=None):
        self.db_name = db_name
        self.table_name = table_name
        self.conn = pymysql.connect("localhost", "root", "123456", db_name) if conn is None else conn
        self.cursor = self.conn.cursor()
        sql = """
                        CREATE TABLE IF NOT EXISTS %s (
                            ID INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
                            POINT VARCHAR(255) NOT NULL,
                            USED INT NOT NULL
                        )CHARSET=utf8;
                        """ % table_name
        try:
            self.cursor.execute(sql)
        except Exception as e:
            print(e)

        self.points = [(point,0) for point in points]
        sql = """
                    INSERT INTO points(
                        POINT,USED
                    ) VALUES (%r,%r)
                """
        self.cursor.executemany(sql,self.points)
        self.conn.commit()


    def pop(self):
        # 提取之后将used字段置为1
        self.cursor.execute("""
                        SELECT POINT FROM %s WHERE USED=0 LIMIT 1
                        """ % self.table_name)
        c = self.cursor.fetchone()
        if c:
            p = c[0]  # fetchone会取出一个列表
            self.cursor.execute("""
                                    UPDATE %s SET USED=1 WHERE POINT=%r
                                    """ % (self.table_name, p))
            self.conn.commit()
        else:
            p = None
        return p


class baidu_spider:
    def __init__(self,ak='f8QB8NM61oZ2j1FuotFa4z9PB9lUqRwE',keyword='美食',city='合肥',density=150,db_name = "point_queue",table_name = 'food_data',radius = 500,output = 'json',scope = 2,conn=None):
        self.ak = ak
        self.keyword = keyword
        self.city = city
        self.density = density
        self.radius = radius
        self.scope = scope
        self.output = output
        self.conn = pymysql.connect("localhost", "root", "123456", db_name,charset='utf8') if conn is None else conn
        self.cursor = self.conn.cursor()
        sql = """
                                CREATE TABLE IF NOT EXISTS %s (
                                    ID INT NOT NULL AUTO_INCREMENT,
                                    NAME VARCHAR(255) NOT NULL,
                                    TAG VARCHAR(255),
                                    RATING VARCHAR(255),
                                    DISTRICT VARCHAR(255),
                                    COORDINATE VARCHAR(255) NOT NULL,
                                    COMMENT_NUM VARCHAR(255),
                                    ADDRESS VARCHAR(255),
                                    TYPE VARCHAR(255),
                                    PRIMARY KEY (ID,NAME,COORDINATE)
                                )CHARSET=utf8;
                                """ % table_name
        try:
            self.cursor.execute(sql)  # 将url设置为主键，便不能重复
        except Exception as e:
            print(e)
        self.table_name = table_name


    def generate_points(self):
        grid = GRID[self.city]
        lat = ['%r' % (grid[0][0] + (grid[2][0] - grid[0][0]) * i / self.density) for i in range(self.density - 1)]
        lng = ['%r' % (grid[2][1] + (grid[0][1] - grid[2][1]) * i / self.density) for i in range(self.density - 1)]
        self.points = [n+','+a for a in lat for n in lng]
        self.queue = MysqlQueue(points=self.points)

    def get_coordinate(self):
        self.generate_points()
        p = self.queue.pop()
        i = 0
        while p:
            if i % 100 == 0:
                print(i)
            i += 1
            url = 'http://api.map.baidu.com/place/v2/search?query=%s&location=%s' \
                    '&radius=%d&output=%s&scope=%s&ak=%s' % (self.keyword,p,self.radius,self.output,self.scope,self.ak)
            r = requests.get(url)
            information = json.loads(r.text)['results']
            if information != []:
                for infor in information:
                    name = infor['name']
                    coordinate = str(infor['location']['lng']) +','+ str(infor['location']['lat'])
                    district = infor['area']
                    address = infor['address']

                    if infor['detail'] == 1:
                        try:
                            tag = infor['detail_info']['tag']
                        except Exception as e:
                            tag = ""
                        try:
                            is_type = infor['detail_info']['type']
                        except Exception as e:
                            is_type = ""

                        try:
                            rating = infor['detail_info']['overall_rating']
                        except Exception as e:
                            rating = ""
                        try:
                            comment_num = infor['detail_info']['comment_num']
                        except Exception as e:
                            comment_num = ""

                    else:
                        tag,is_type,rating,comment_num = "","","",""

                    sql = """
                    INSERT INTO %s(
                        NAME,TAG,RATING,DISTRICT,
                        COORDINATE,COMMENT_NUM,ADDRESS,TYPE
                    ) VALUES (%r,%r,%r,%r,%r,%r,%r,%r) ON DUPLICATE KEY UPDATE NAME=%r;
                    """ % (self.table_name,name,tag,rating,district,coordinate,comment_num,address,is_type,name)
                    self.cursor.execute(sql)
                    self.conn.commit()
            p = self.queue.pop() # 重新获取下一个p


if __name__ == "__main__":
    bs = baidu_spider(density=150)
    bs.get_coordinate()

