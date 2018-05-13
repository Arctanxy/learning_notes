from flask import render_template
from . import auth
from flask.ext.login import logout_user,login_required
from flask import Blueprint,render_template,redirect,request,url_for,flash
from flask.ext.login import login_user
from . import auth
from ..models import User 
from .forms import LoginForm

@auth.route('/login',methods=['GET','POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user is not None and user.verify_password(form.password.data):
            login_user(user,form.remember_me.data)
            return redirect(request.args.get('next') or url_for('main.index'))
        flash('Invalid username or password')
    return render_template('auth/login.html',form=form)

@auth.route('/logout')
def logout():
    logout_user()
    flash('You have been logged out')
    return redirect(url_for('main.index'))