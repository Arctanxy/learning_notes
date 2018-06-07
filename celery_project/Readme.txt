# Celery环境配置

1. 安装celery

	pip install celery==3.1.25
	
2. 安装RabbitMQ
	
	https://blog.csdn.net/lu1005287365/article/details/52315786
	
3. 配置MongoDB

	1. 额外安装celery的MongoDB模块
		
		pip install -U celery[mongodb]
		
	2. 设置BROKER_URL和BACK_END
		
		BROKER_URL = 'mongodb://localhost:27017/database_name'
		
		BACKEND:http://docs.celeryproject.org/en/3.1/configuration.html#conf-mongodb-result-backend
	
	