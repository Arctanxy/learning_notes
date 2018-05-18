# Pytorch 迁移学习

https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#sphx-glr-beginner-transfer-learning-tutorial-py



迁移学习可以理解为把别人训练好的模型结构与参数拿来，在自己的数据上训练，得到最终模型。因为很多项目中的数据量都不够大，采用这种方式既能节省时间，又可以提高准确率。

在这个教程中，你将学习如何使用迁移学习来训练神经网络，关于迁移学习，[斯坦福大学的CS231n课程笔记中有较详细的描述](http://cs231n.github.io/transfer-learning/)

> 实际上很少有人会从头开始训练整个卷积神经网络（从随机初始化的参数开始训练），因为很少有人有足够大的数据集。通常都会选择使用一个在足够大的数据集（比如ImageNet）上预先训练过的卷积网络，然后把这个神经网络作为初始模型或者特征提取器带入到实际项目中。

**微调卷积网络**：使用在ImageNet数据集中预训练过的神经网络，在此基础上进行训练。

**将卷积网络作为特征提取器**：将卷积网络中除了全连接层以外的层的参数固定下来，全连接层用随机初始化参数的全连接层替代，只训练新的全连接层。

需要工具：

```python
from __future__ import print_function ,division

import torch
import torch.nn as nn 
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np 
import torchvision
from torchvision import datasets,models,transforms
import matplotlib.pyplot as plt 
import time
import os
import copy

plt.ion()  # 交互模式
```

我们将使用torchvision 和 torch.utils.data 来加载数据

这篇教程中要完成的任务是训练一个能区分蜜蜂和蚂蚁的模型。我们有大约120张训练图片，每个类别下有75张验证图片。如果从头开始训练神经网络的话，这些数据实在是太少了。如果使用迁移学习的话，也许能获得比较好的结果。

[下载地址](https://download.pytorch.org/tutorial/hymenoptera_data.zip)

原始数据加载过程：

```python
data_transforms = {
    'train':transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    'val':transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
}
data_dir = 'hymenoptera_data'

image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ['train','val']}

dataloaders = {x:torch.utils.data.DataLoader(image_datasets[x],batch_size=4,shuffle = True,num_works = 4) for x in ['train','val']}

dataset_sizes = {x:len(image_datasets[x]) for x in ['train','val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

展示图片分类

```python
def imshow(inp,title = None):
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    inp = std * inp + mean
    inp = np.clip(inp,0,1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

inputs,classes = next(iter(dataloader['train']))

out = torchvision.utils.make_grid(inputs)

imshow(out,title=[class_names[x] for x in classes])
```

