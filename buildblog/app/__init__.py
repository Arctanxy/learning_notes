from flask.ext.login import LoginManager

login_manager = LoginManager()
login_manager.session_protection = 'strong'
login_manager.login_view = 'auth.login'

def create_app(config_name):
    #附加蓝本
    from .auth import auth as auth_blueprint
    login_manager.init_app(app)
    app.register_blueprint(auth_blueprint,url_prefix='/auth')

    return app
