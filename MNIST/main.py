import os
import pandas as pd
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt
import torch as t
from tqdm import tqdm
from torch.autograd import Variable
from mnist_models import conv_net,to_image,fc_net,AlexNet
import signal

# 设置模型参数
TYPE = 'cla'
METHOD = 'res'
EPOCHS = 1
BATCH_SIZE = 5
LR = 0.001

# 读取数据
train = pd.read_csv('./data/train.csv')
data = train.drop('label',axis=1)
test = pd.read_csv('./data/test.csv')
test_data = t.from_numpy(test.values).float()
data = data.values

# 标签与自变量处理
y = train['label'].values
y = t.from_numpy(y).long()
data = t.from_numpy(data).float()
data,y = Variable(data),Variable(y)

# 初始化模型
if METHOD == 'conv':
    data = to_image(data) # 将数据转为二维
    test_data = to_image(test_data)
    net = conv_net()
elif METHOD == 'fc':
    net = fc_net()
elif METHOD == 'res':
    # 使用resnet18进行迁移学习，微调参数，如果冻结参数，将resnet作为特征选择器的话，训练速度更快。
    # 因为resnet参数过多，不建议使用CPU运算，使用Xeon E5620一个EPOCH要训练三个小时
    data = to_image(data)
    test_data = to_image(test_data)
    net = models.resnet18(pretrained=True)
    # 固定参数
    for p in net.parameters():
        p.requires_grad = False

    # 因为MNIST图片是单通道，并且尺寸较小，所以需要对resnet进行一些细节修改
    net.conv1 = t.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=3,
                           bias=False)
    net.maxpool = t.nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
    net.avgpool = t.nn.AvgPool2d(5, stride=1)

    num_ftrs = net.fc.in_features
    net.fc = t.nn.Linear(num_ftrs,10)

elif METHOD == 'alex':
    data = to_image(data)
    test_data = to_image(test_data)
    net = AlexNet()

else:
    raise Exception("Wrong Method!")


# 如果模型文件存在则尝试加载模型参数
if os.path.exists('H:/learning_notes/MNIST/%s.pth' % METHOD):
    try:
        net.load_state_dict(t.load('H:/learning_notes/MNIST/%s.pth' % METHOD))
    except Exception as e:
        print(e)
        print("Parameters Error")

# 定义模型代价函数
if TYPE == 'reg':
    criterion = t.nn.MSELoss()
elif TYPE == 'cla':
    criterion = t.nn.CrossEntropyLoss()
else:
    raise Exception("Wrong Type!")

# 定义优化器
if METHOD == 'res':
    # 如果是用的resnet，则只训练最后的全连接层的参数
    optim = t.optim.Adam(net.fc.parameters(),lr = 0.001,weight_decay=0.0)
else:
    optim = t.optim.Adam(net.parameters(),lr=0.001,weight_decay=0.0)


# plt.ion() # 用于绘制动态图
# losses = []

# 用于捕捉KeyboardInterrupt错误，效果比try except好得多
# 可以人为终止训练，并将训练得到的参数保存下来，实现断点训练
def exit(signum, frame):
    print("Model Saved")
    t.save(net.state_dict(), 'H:/learning_notes/MNIST/%s.pth' % METHOD)
    raise KeyboardInterrupt

signal.signal(signal.SIGINT, exit)
signal.signal(signal.SIGTERM, exit)


# 开始训练
for epoch in tqdm(range(EPOCHS)):
    index = 0
    if epoch % 100 == 0:
        for param_group in optim.param_groups:
            LR = LR * 0.9
            param_group['lr'] = LR
    for i in tqdm(range(int(len(data)/BATCH_SIZE)),total=int(len(data)/BATCH_SIZE)):

        batch_x = data[index:index + BATCH_SIZE]
        batch_y = y[index:index + BATCH_SIZE]
        prediction = net.forward(batch_x)
        loss = criterion(prediction, batch_y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        index += BATCH_SIZE  # 进入下一个batch
        # if loss <= 0.3:
            # losses.append(loss)
        # plt.plot(losses)
        # plt.pause(0.001)

        print(loss)
t.save(net.state_dict(),'H:/learning_notes/MNIST/%s.pth' % METHOD)
# plt.ioff()
submission = pd.read_csv("./data/sample_submission.csv")

print('=======Predicting========')

# 切换成验证模式，验证模式下DROPOUT将不起作用
net.eval()

test_data = Variable(test_data)

result = t.Tensor()

index = 0

# 分段进行预测，节省内存
for i in tqdm(range(int(test_data.shape[0]/BATCH_SIZE)),total=int(test_data.shape[0]/BATCH_SIZE)):
    label_prediction = net(test_data[index:index+BATCH_SIZE])
    index += BATCH_SIZE
    result = t.cat((result,label_prediction),0)

# 结果处理
if TYPE == 'cla':
    _,submission['Label'] = t.max(result.data,1) # t.max返回一个元祖，第一个元素是最大元素值，第二个元素是最大元素位置
elif TYPE == 'reg':
    submission['Label'] = submission['Label'].astype('int')
    submission['Label'] = submission['Label'].apply(lambda x:9 if x>= 10 else x)


submission.to_csv("submission.csv",index=False)