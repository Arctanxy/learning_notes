import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import collections

class k_means(object):
    def __init__(self,k = 4,max_times= 1000):
        self.k = k
        self.kpoints = []
        self.x_dict = collections.defaultdict(list)
        self.max_times = max_times

    def fit(self,x):
        #先随机生成k个点
        self.kpoints = np.random.random((self.k,x.shape[1]))
        
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
                self.kpoints[i] = np.mean(self.x_dict[i],axis=1)

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
        return i

    def show_labels(self):
        for i in range(self.k):
            for m,samples in self.x_dict[i]:
                for p in samples:
                    plt.scatter(p)
        plt.show()


if __name__ == "__main__":
    np.random.seed(10)
    data = np.random.random((100,2))

    clf = k_means()
    clf.fit(data)
    clf.show_labels()
