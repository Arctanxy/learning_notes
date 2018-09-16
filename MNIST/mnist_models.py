import torch as t
def to_image(data):
    data = data.view(-1,1,28,28)
    return data

class fc_net(t.nn.Module):
    '''
    全连接网络
    '''
    def __init__(self):
        super(fc_net,self).__init__()
        self.fc1 = t.nn.Sequential(t.nn.Linear(784,200),t.nn.ReLU())
        self.fc2 = t.nn.Sequential(t.nn.Linear(200,100),t.nn.ReLU())
        self.fc3 = t.nn.Sequential(t.nn.Linear(100,20),t.nn.ReLU())
        self.fc4 = t.nn.Linear(20,10)
    def forward(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

class conv_net(t.nn.Module):
    '''
    卷积网络，需先将数据转为2维图片形式
    '''
    def __init__(self):
        super(conv_net,self).__init__()
        self.conv1 = t.nn.Sequential(
            t.nn.Conv2d(1,10,5,1,1),
            t.nn.MaxPool2d(2),
            t.nn.ReLU(),
            t.nn.BatchNorm2d(10)
        )
        self.conv2 = t.nn.Sequential(
            t.nn.Conv2d(10,20,5,1,1),
            t.nn.MaxPool2d(2),
            t.nn.ReLU(),
            t.nn.BatchNorm2d(20) # num_features为通道数
        )
        self.fc1 = t.nn.Sequential(
            t.nn.Linear(500,60),
            t.nn.Dropout(0.5),
            t.nn.ReLU()
        )
        self.fc2 = t.nn.Sequential(
            t.nn.Linear(60,20),
            t.nn.Dropout(0.5),
            t.nn.ReLU()
        )
        self.fc3 = t.nn.Linear(20,10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1,500)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class AlexNet(t.nn.Module):
    '''
    类似AlexNet的神经网络，因为电脑配置及MNIST数据集图片尺寸问题，将Kernel_size和stride都改小了
    '''
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = t.nn.Sequential(
            t.nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            t.nn.ReLU(inplace=True),
            t.nn.MaxPool2d(kernel_size=3, stride=1),
            t.nn.Conv2d(64, 192, kernel_size=3, padding=2),
            t.nn.ReLU(inplace=True),
            t.nn.MaxPool2d(kernel_size=3, stride=2),
            t.nn.Conv2d(192, 384, kernel_size=3, padding=1),
            t.nn.ReLU(inplace=True),
            t.nn.Conv2d(384, 256, kernel_size=3, padding=1),
            t.nn.ReLU(inplace=True),
            t.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            t.nn.ReLU(inplace=True),
            t.nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = t.nn.Sequential(
            t.nn.Dropout(),
            t.nn.Linear(256 * 6 * 6, 4096),
            t.nn.ReLU(inplace=True),
            t.nn.Dropout(),
            t.nn.Linear(4096, 4096),
            t.nn.ReLU(inplace=True),
            t.nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        print(x.shape)
        x = x.view(x.size(0), 256 * 6 * 6)
        print(x.shape)
        x = self.classifier(x)
        return x