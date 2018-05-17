# Pytorch 迁移学习

https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#sphx-glr-beginner-transfer-learning-tutorial-py



迁移学习可以理解为把别人训练好的模型结构与参数拿来，在自己的数据上训练，得到最终模型。因为很多项目中的数据量都不够大，采用这种方式既能节省时间，又可以提高准确率。

在这个教程中，你将学习如何使用迁移学习来训练神经网络，关于迁移学习，[斯坦福大学的CS231n课程笔记中有较详细的描述](http://cs231n.github.io/transfer-learning/)

> 实际上很少有人会从头开始训练整个卷积神经网络（从随机初始化的参数开始训练），因为很少有人有足够大的数据集。通常都会选择使用一个在足够大的数据集（比如ImageNet）上预先训练过的卷积网络，然后把这个神经网络作为初始模型或者特征提取器带入到实际项目中。

