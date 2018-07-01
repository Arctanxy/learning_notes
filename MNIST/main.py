import torch as t
import torch.nn.functional as F
import pandas as pd
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from config import options
import numpy as np

options = options()

class Net(t.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        if options.method == "Square":
            self.conv1 = t.nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
            self.conv2 = t.nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
            self.fc1 = t.nn.Linear(16*20*20,200)
            self.fc2 = t.nn.Linear(200,100)
            self.predict = t.nn.Linear(100,10)
        elif options.method == 'Linear':
            self.hidden1 = t.nn.Linear(784,200)
            self.hidden2 = t.nn.Linear(200,100)
            self.predict = t.nn.Linear(100,10)
        else:
            raise Exception("Wrong Method")

    def forward(self, x):
        if options.method == 'Square':
            x = F.max_pool2d(self.conv1(x),(2,2))
            x = F.max_pool2d(self.conv2(x),2)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.softmax(self.predict(x))
        elif options.method == 'Linear':
            x = F.relu(self.hidden1(x))
            x = F.relu(self.hidden2(x))
            x = F.softmax(self.predict(x))
        return x

def train_data():
    train_data = pd.read_csv('data/train.csv')

    y = train_data['label'].values
    x = train_data.drop('label', axis=1).values
    if options.method == 'Linear':
       pass
    elif options.method == 'Square':
        x = np.array([a.reshape(28,28) for a in x])
    x, y = t.from_numpy(x), t.from_numpy(y)
    x = x.type(t.FloatTensor)
    y = y.type(t.LongTensor)
    x, y = Variable(x), Variable(y)
    return x, y




if __name__ == "__main__":
    net = Net()
    print(net)
    x,y = train_data()
    print(x.shape)
    criterion = t.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),lr=0.01,momentum = 0.9)
    for epoch in range(1000):
        for i,(data,labels)
        prediction = net(x)
        loss = criterion(prediction,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(loss.data[0])
