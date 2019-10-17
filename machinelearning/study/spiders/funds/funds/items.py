# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class FundsItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()

    code = scrapy.Field()   # 基金代码
    name = scrapy.Field()   # 基金名称
    unitNetWorth = scrapy.Field()   # 单位净值
    day = scrapy.Field()    # 日期
    dayOfGrowth = scrapy.Field()  # 日增长率
    recent1Week = scrapy.Field()    # 最近一周
    recent1Month = scrapy.Field()   # 最近一月
    recent3Month = scrapy.Field()   # 最近三月
    recent6Month = scrapy.Field()   # 最近六月
    recent1Year = scrapy.Field()    # 最近一年
    recent2Year = scrapy.Field()    # 最近二年
    recent3Year = scrapy.Field()    # 最近三年
    fromThisYear = scrapy.Field()   # 今年以来
    fromBuild = scrapy.Field()  # 成立以来
    serviceCharge = scrapy.Field()  # 手续费
    upEnoughAmount = scrapy.Field()     # 起购金额

    pass
