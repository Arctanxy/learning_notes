# 使用flask搭建个人博客

需要扩展：

1. Flask-Login:管理已登录用户的会话
2. Werkzeug:计算密码哈希值并进行核对
3. itsdangerous:生成并核对加密安全令牌
4. Flask-Mail:发送与认证相关的电子邮件
5. Flask-Bootstrap:HTML模板
6. Flask-WTF:Web表单

## 使用WerkZeug实现密码散列


```python

from werkzeug.security import generate_password_hash,check_password_hash

class User(db.Model):
    #..
    password_hash = db.Column(db.String(128))

    @property
    def password(self):
        raise AttributeError('password is not a readable attribute') # 这是一个只写属性，不允许读取

    @password.setter
    def password(self,password):
        self.password_hash = generate_password_hash(password)
    
    def verify_password(self,password):
        return check_password_hash(self.password_hash,password) # 将密码与库中的密码进行比对，判断是否正确

```
**测试** 
```python
import unittest
from app.models import User

class UserModelTestCase(unittest.TestCase):
    # 测试设置密码的功能是否有效
    def test_password_setter(self):
        u = User(password='cat')
        self.assertTrue(u.password_hash is not None)

    # 测试密码是不是一个只写属性
    def test_no_password_getter(self):
        u = User(password='cat')
        with self.assertRaises(AttributeError):
            u.password
    
    # 测试密码验证函数的功能是否正常
    def test_password_verification(self):
        u = User(password='cat')
        self.assertTrue(u.verify_password('cat'))
        self.assertFalse(u.verify_password('dog'))

    # 测试不同用户设置相同密码后生成不同哈希码的功能是否正常
    def test_password_salts_are_random(self):
        u = User(password='cat')
        u2 = User(password='cat')
        self.assertTrue(u.password_hash != u2.password_hash)

```

## 创建认证蓝本

