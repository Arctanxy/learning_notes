# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html

import csv

class InfluentPipeline(object):

    def process_item(self, item, spider):
        with open('D:/Modeling/ANHUI/HANSHANXIAN/items.csv','a',encoding='utf-8') as f:

            csv.writer(f).writerow((
                'is_type','name','coord','district','address'
            ))
        return item