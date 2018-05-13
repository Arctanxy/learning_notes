# 博客主文件
import datetime
import functools
import os
import re 
import urllib

from flask import (Flask,abort,flash,Markup,redirect,render_template,
                request,Response,session,url_for)

from markdown import markdown
from markdown.extensions.codehilite import CodeHiliteExtension # 文章中代码高亮
from markdown.extensions.extra import ExtraExtension
from micawber import bootstrap_basic,parse_html
from micawber.cache import Cache as OEmbedCache
from peewee import *
from playhouse.flask_utils import FlaskDB,get_object_or_404,object_list
from playhouse.sqlite_ext import *

ADMIN_PASSWORD = 'secret'
APP_DIR = os.path.dirname(os.path.realpath(__file__))
DATABASE = 'sqliteext:///%s' % os.path.join(APP_DIR,'blog.db')
DEBUG = False
SECRET_KEY = 'this is a secret' # 用于Flask加密会话Cookie
SITE_WIDTH = 800

app = Flask(__name__)
app.config.from_object(__name__)

flask_db = FlaskDB(app)
database = flask_db.database

oembed_providers = bootstrap_basic(OEmbedCache())

# 数据库操作,将使用SQLite数据库
class Entry(flask_db.Model):
    
    # 库中每篇文章将包含如下五条信息
    title = CharField()
    slug = CharField(unique=True)
    content = TextField()
    publisthed = BooleanField(index=True)
    timestamp = DateTimeField(default=datetime.datetime.now,index=True)

    @property
    def html_content(self): # 将markdown转为html并处理视频连接
        hilite = CodeHiliteExtension(linenums = False,css_class='highlight')
        extras = ExtraExtension()
        markdown_content = markdown(self.content,extensions = [hilite,extras])
        oembed_content = parse_html(
            markdown_content,
            oembed_providers,
            urlize_all=True,
            maxwidth=app.config['SITE_WIDTH'])
        return Markup(oembed_content)

    def save(self,*args,**kwargs):
        if not self.slug:
            self.slug = re.sub('[^\w]+','-',self.title.lower()) 
    
        ret = super(Entry,self).save(*args,**kwargs)

        # 保存内容
        self.update_search_index()
        return ret

    def update_search_index(self):
        search_content = '\n'.join((self.title,self.content))
        try:
            fts_entry = FTSEntry.get(FTSEntry.docid == self.id)
        except FTSEntry.DoesNotExist:
            FTSEntry.create(docid=self.id,content=search_content)
        else:
            fts_entry.content = search_content
            fts_entry.save()

    @classmethod
    def public(cls):
        return Entry.select().where(Entry.publisthed == True)
    
    @classmethod
    def search(cls,query):
        words = [word.strip() for word in query.split() if word.strip()]
        if not words:
            return Entry.select().where(Entry.id == 0)
        else:
            search = ''.join(words)
        
        return (Entry.select(Entry,FTSEntry.rank().alias('score'))
                .join(FTSEntry,on=(Entry.id == FTSEntry.docid))
                .where((Entry.publisthed == True) & (FTSEntry.match(search)))
                .order_by(SQL('score')))

    @classmethod
    def drafts(cls):
        return Entry.select().where(Entry.publisthed == False)

class FTSEntry(FTSModel):
    content = SearchField()

    class Meta:
        database = database

# 登入登出
def login_required(fn):
    @functools.wraps(fn)
    def inner(*args,**kwargs):
        if session.get('logged_in'):
            return fn(*args,**kwargs)
        return redirect(url_for('login',next=request.path))
    return inner

@app.route('/login/',methods=['GET','POST'])
def login():
    next_url = request.args.get('next') or request.form.get('next')
    if request.method == 'POST' and request.form.get('password'):
        password = request.form.get('password')
        if password == app.config['ADMIN_PASSWORD']:
            session['logged_in'] = True
            session.permanent = True # 使用Cookie存储会话信息
            flash('You are now Logged in.','success')
            return redirect(next_url or url_for('index'))
        else:
            flash('Incorrect password.','danger')
    return render_template('login.html',next_url=next_url)

@app.route('/logout/',methods=['GET','POST'])
def logout():
    if request.method == 'POST':
        session.clear()
        return redirect(url_for('login'))
    return render_template('logout.html')

# 视图函数
@app.route('/')
def index():
    search_query = request.args.get('q')
    if search_query:
        query = Entry.search(search_query)
    else:
        query = Entry.public().order_by(Entry.timestamp.desc()) # 按时间顺序排列
    return object_list('index.html',query,search = search_query,check_bounds=False)

@app.route('/drafts/') #草稿
@login_required
def drafts():
    query = Entry.drafts().order_by(Entry.timestamp.desc())
    return object_list('index.html',query)

@app.route('/create/',methods=['GET','POST'])
@login_required
def create():
    if request.method == 'POST':
        if request.form.get('title') and request.form.get('content'):
            entry = Entry.create(
                title = request.form['title'],
                content = request.form['content'],
                published = request.form.get('published') or False
            )
            if entry.publisthed:
                return redirect(url_for('detail',slug=entry.slug))
            else:
                return redirect(url_for('edit',slug = entry.slug))
        else:
            flash('Title and Content are required','danger')
    return render_template('create.html')

@app.route('/<slug>/') # 用于生成包含文章标题名的url
def detail(slug):
    if session.get('logged_in'):
        query = Entry.select()
    else:
        query = Entry.public() # 非登录用户只显示公开文章
    entry = get_object_or_404(query,Entry.slug == slug)
    return render_template('detail.html',entry = Entry)

@app.route('/<slug>/edit/',methods = ['GET','POST'])
@login_required
def edit(slug):
    entry = get_object_or_404(Entry,Entry.slug == slug)
    if request.method == 'POST':
        if request.form.get('title') and request.form.get('content'):
            entry.title = request.form['title']
            entry.content = request.form['content']
            entry.published = request.form.get('published') or False
            entry.save()

            flash('Entry saved successfully','sucess')
            if entry.published:
                return redirect(url_for('detail',slug = entry.slug))
            else:
                return redirect(url_for('edit',slug=entry.slug))
        else:
            flash('Title and Content are required','danger')
    
    return render_template('edit.html',entry=entry)

# 错误处理
@app.template_filter('clean_querystring')
def clean_querystring(request_args,*keys_to_remove,**new_values):
    querystring = dict((key,value) for key,value in request_args.items())
    for key in keys_to_remove:
        querystring.pop(key,None)
    querystring.update(new_values)
    return urllib.urlencode(querystring)

@app.errorhandler(404)
def not_found(exc):
    return Response('<h3> Not Found </h3>'),404

def main():
    database.create_tables([Entry,FTSEntry],safe= True)
    app.run(debug = True)

if __name__ == "__main__":
    main()
    