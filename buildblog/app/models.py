from werkzeug.security import generate_password_hash,check_password_hash
from flask.ext import UserMixin
from . import login_manager

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(UserMixin,db.Model):
    #..
    __tablename__ = 'users'
    id = db.Column(db.Interger,primary_key = True)
    email = db.Column(db.String(64),unique=True,index=True)
    username = db.Column(db.String(64),unique=True,index=True)
    password_hash = db.Column(db.String(128))
    role_id = db.Column(db.Interger,db.ForeignKey('roles.id'))

    @property
    def password(self):
        raise AttributeError('password is not a readable attribute') # 这是一个只写属性，不允许读取

    @password.setter
    def password(self,password):
        self.password_hash = generate_password_hash(password)
    
    def verify_password(self,password):
        return check_password_hash(self.password_hash,password) # 将密码与库中的密码进行比对，判断是否正确


