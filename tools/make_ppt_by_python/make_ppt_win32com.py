import win32com.client
# import win32com.gen_py.ppt11b as ppt11b
import requests as req
import re
from lxml import html
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.87 Safari/537.36',
}
r = req.get("https://www.jianshu.com/p/2ff1ead2249a",headers=headers)
tree = html.fromstring(r.text)
# title = re.findall(r'<div .*?>(.*?)</div>',r.text)
content = tree.xpath('//h1|//h2|//h4|//p//img/@data-original-src')

ppt = win32com.client.Dispatch("Powerpoint.Application")
ppt.Visible = True # 修改完毕后
pptfile = ppt.Presentations.Open('H:/learning_notes/make_ppt_by_python/template.pptx',ReadOnly=0,Untitled=0,WithWindow=1)

Slide1 = pptfile.Slides.Add(len(pptfile.Slides)+1,11) # 添加标题
Slide1.Shapes.Title.TextFrame.TextRange.Text = content[0].text
Slide2 = pptfile.Slides.Add(len(pptfile.Slides)+1,2) # 前言
Slide2.Shapes.Title.TextFrame.TextRange.Text = "简介"
# shape1 = Slide1.Shapes.AddTextbox(Orientation=0x1,Left=100,Top=50,Width=400,Height=100)
# shape1.TextFrame.TextRange.Text = content[1].text + '\n' + content[2].text
shape = Slide2.Placeholders(2)
shape.TextFrame.TextRange.Text = content[1].text + '\n' + content[2].text
for c in content[3:]:
    print(c.text)
    try:
        if '</h2>' in html.tostring(c).decode('utf-8'):
            Slide = pptfile.Slides.Add(len(pptfile.Slides)+1,2)
            Slide.Shapes.Title.TextFrame.TextRange.Text = c.text
        # 如果是4级标题，则另起一个空白页
        elif '</h4>' in html.tostring(c).decode('utf-8'):
            Slide = pptfile.Slides.Add(len(pptfile.Slides)+1,2)
            Slide.Shapes.Title.TextFrame.TextRange.Text = c.text
        # 如果是段落内容，则添加到文本框中
        elif '</p>' in html.tostring(c).decode('utf-8'):
            shape = Slide.Placeholders(2)
            shape.TextFrame.TextRange.Text += c.text
    except Exception as e:
        pass
    
    
# shape2 = Slide.Shapes.AddPicture(FileName='H:/learning_notes/make_ppt_by_python/positionLabel.png',LinkToFile=False,SaveWithDocument=True,Left=100,Top=100,Width=100,Height=400)
pptfile.Save()
pptfile.Close()