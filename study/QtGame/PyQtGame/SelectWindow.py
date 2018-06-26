from PyQt5.QtWidgets import QApplication,QPushButton,QWidget,QLabel,QLineEdit,QCheckBox
import PyQt5.QtCore as Qt

class SelectWindow(QWidget):
    def __init__(self):
        super(SelectWindow,self).__init__()
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle("test")
        self.setGeometry(400,400,300,300)
        # 输出框
        self.output_label1 = QLabel(self)
        self.output_label1.setGeometry(100,100,300,20)
        self.output_label1.setText("请输入待预测列标")
        # 单行输入文本框
        self.input_label1 = QLineEdit(self)
        self.input_label1.setGeometry(100,150,250,20)
        # 输出框
        self.output_label2 = QLabel(self)
        self.output_label2.setGeometry(100,200,300,20)
        self.output_label2.setText("请输入自变量列表(使用英文逗号连接)")
        # 单行输入文本框
        self.input_label2 = QLineEdit(self)
        self.input_label2.setGeometry(100,250,250,20)
        # 确定按钮
        self.yes_button = QPushButton(self)
        self.yes_button.clicked.connect(self.yes)
        self.yes_button.setText("确认")
        self.yes_button.move(50,50)
        # 取消按钮
        self.no_button = QPushButton(self)
        self.no_button.clicked.connect(self.no)
        self.no_button.setText("取消")
        self.no_button.move(150,50)
    
    def yes(self):
        self.target = self.input_label1.text()
        self.argument = self.input_label2.text()
        self.close()

    def no(self):
        self.target = None
        self.argument = None
        self.close()
        return None