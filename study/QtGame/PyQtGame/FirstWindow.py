from PyQt5.QtWidgets import QHBoxLayout,QApplication,QPushButton,QTableWidget,QWidget,QLabel,QLineEdit,QCheckBox,QTableWidgetItem
import PyQt5.QtCore as Qt
import sys
import pandas as pd 
import numpy as np 
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from SelectWindow import SelectWindow
from TableWindow import TableWindow

class FirstWindow(QWidget):

    def __init__(self):
        super(FirstWindow,self).__init__()
        self.initUI()
        

    def initUI(self):
        self.son_window = SelectWindow()
        self.tablewindow = TableWindow()
        self.setWindowTitle("test")
        self.setGeometry(400,400,800,800)
        # 按钮
        self.first_button = QPushButton(self)
        self.first_button.setText("读取文档")
        self.first_button.clicked.connect(self.read_data)
        self.first_button.move(50,50)
        # 选择按钮
        self.select_button = QPushButton(self)
        self.select_button.setText("选择列")
        self.select_button.clicked.connect(self.son_window.show)
        self.select_button.move(150,50)
        # 显示数据按钮
        self.show_button = QPushButton(self)
        self.show_button.setText("显示数据")
        self.show_button.clicked.connect(self.show_data)
        self.show_button.move(250,50)

        # 关闭按钮
        self.exit_button = QPushButton(self)
        self.exit_button.setText("关闭程序")
        self.exit_button.clicked.connect(self.close)
        self.exit_button.move(350,50)

        # 输出框
        self.output_label = QLabel(self)
        self.output_label.setGeometry(100,150,300,300)
        # 单行输入文本框
        self.input_label = QLineEdit(self)
        self.input_label.setText("请输入文件路径")
        self.input_label.setGeometry(100,100,250,50)
        
        # 建立模型
        self.built_button = QPushButton(self)
        self.built_button.setText("建立模型")
        self.built_button.clicked.connect(self.build_model)
        self.built_button.move(350,50)

    def read_data(self):
        # 读取数据
        try:
            self.df = pd.read_excel(str(self.input_label.text()))
            self.cols = self.df.columns
            self.output_label.setText('\n'.join(self.cols))
        except Exception as e:
            self.output_label.setText(str(e))
            return None

    def build_model(self):
        # 从子窗口中获取自变量和因变量
        target = self.son_window.target
        argument = self.son_window.argument
        # 分割自变量
        arguments = argument.split(',')
        x = self.df[arguments]
        y = self.df[target]
        # 如果只有一个自变量，需要增加维度
        if len(x.shape) == 1:
            x = x.reshape(-1,1)
        # 建立模型
        clf = Ridge()
        clf.fit(x,y)
        coefs = clf.coef_ 
        intercepts = clf.intercept_
        self.output_label.setText("系数：%s;截距：%f" % (str(coefs),intercepts))

    def show_data1(self):
        self.tablewindow.mytable = QTableWidget(5,len(self.cols))
        self.tablewindow.mytable.setHorizontalHeaderLabels(self.cols)
        for i in range(5):
            for j in range(len(self.cols)):
                self.tablewindow.mytable.setItem(i,j,QTableWidgetItem(str(self.df[self.cols[j]].values[i])))
        # 数据显示
        layout = QHBoxLayout()  
        layout.addWidget(self.tablewindow.mytable)  
        self.tablewindow.setLayout(layout) 

    def show_data(self):
        self.show_data1()
        self.tablewindow.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FirstWindow()
    window.show()
    sys.exit(app.exec_())