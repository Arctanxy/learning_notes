import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import collections

class k_means(object):
    def __init__(self,k = 4,max_times= 100):
        self.k = k
        self.kpoints = []
        self.x_dict = collections.defaultdict(list)
        self.max_times = max_times

    def fit(self,x):
        #先随机生成k个点，也不能太随机，要取在数据分布区，否则可能导致某个类无样本
        self.kpoints = np.array([np.mean(x,axis=0)+ 0.1 * np.random.random((x.shape[1])) for i in range(self.k)])
        plt.ion()
        fig = plt.figure()
        ax  = plt.gca()
        for i in range(self.max_times):
            #初始化x_dict
            self.x_dict = collections.defaultdict(list)
            #分组
            for sample in x:
                for i in range(self.k):
                    if self.cluster(sample) == i:
                        self.x_dict[i].append(sample)
            
            #求新的基准点
            for i in range(self.k):
                self.kpoints[i] = np.mean(self.x_dict[i],axis=0)
            #动态显示分组变化
            if x.shape[1] <= 2:
                self.show_labels(fig,ax)
            print([len(self.x_dict[i]) for i in range(self.k)])

    def distance(self,sample,p):
        return np.sqrt(np.sum(np.square(sample-p)))

    def cluster(self,sample):
        min_dis = np.inf
        label = 0
        for i in range(self.k):
            d = self.distance(sample,self.kpoints[i])
            if d < min_dis:
                min_dis = d
                label = i
        return label

    def show_labels(self,fig,ax):
        np.random.seed(8)
        if self.k > 10 or self.kpoints.shape[1] > 2:
            raise Exception("Not able to show labels!")
        color_set = ['red','green','orange','blue','black','yellow','pink','tomato','brown','grey']
        colors = np.random.choice(color_set,size=self.k)
        for i in range(self.k):
            scatters = self.ploting(self.x_dict[i],ax,color = colors[i])
            scatters.remove()
            plt.pause(0.5)

    def ploting(self,points,ax,color = 'red'):
        for p in points:
            scatters = ax.scatter(p[0],p[1],color = color)
        return scatters
            


if __name__ == "__main__":
    #np.random.seed(10)
    data = np.random.random((300,2))
    clf = k_means()
    clf.fit(data)
    clf.show_labels()
    while True:
        plt.pause(1)
