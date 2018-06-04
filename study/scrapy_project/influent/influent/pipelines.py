# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html

import csv,codecs 

class InfluentPipeline(object):

    def __init__(self):
        self.f = open('H:/learning_notes/study/scrapy_project/items.csv','w',newline='',encoding='utf-8')
        self.writer = csv.writer(self.f)
        
        self.writer.writerow((
                'IS_TYPE','IS_NAME','COORDINATE','XZQHSZ_DM','ADDRESS'
            ))

    def process_item(self, item, spider):
        
        if item['name']:
            self.writer.writerow((
                item['is_type'],item['name'],
                item['coord'],item['district'],
                item['address']
            ))
        return item

    def close_spider(self,spider):
        self.f.close()