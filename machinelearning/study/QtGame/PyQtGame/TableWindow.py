'''
表格界面，用于显示数据
'''
from PyQt5.QtWidgets import QTableWidget,QDialog

class TableWindow(QDialog):
    def __init__(self):
        super(TableWindow,self).__init__()


'''        self.initUI(df)

    def initUI(self,df=None):
        try:
            cols = df.columns
            self.mytable = QTableWidget(5,len(cols))
            self.mytable.setHorizontalHeaderLabels(cols)
            for i in range(5):
                for j in range(len(cols)):
                    self.mytable.setItem(i,j,df[df.columns[j]][i])
        except:
            cols = 5
            self.mytable = QTableWidget(5,cols)
            # self.mytable.setHorizontalHeaderLabels(cols)
'''
