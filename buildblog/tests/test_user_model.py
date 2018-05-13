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
