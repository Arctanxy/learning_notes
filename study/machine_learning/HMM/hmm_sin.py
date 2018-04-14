from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt 
import numpy as np 

n = np.linspace(0,2*np.pi,100)
x = np.sin(n)

diff = np.diff(x).reshape(-1,1)#一维数组记得要转换一下维度，否则会报错
clf = GaussianHMM(n_components=4)
clf.fit(diff)
hidden_states = clf.predict(diff)
colors = ['r','g','b','y']
for i in range(len(x)-1):
    for j in range(clf.n_components):
        if hidden_states[i] == j:
            plt.plot([n[i],n[i+1]],[x[i],x[i+1]],color=colors[j])
new_x,states = clf.sample(50)

for i in range(len(new_x)-1):
    plt.scatter(i,new_x[i],color = 'brown')
    plt.scatter(i,states[i],color = 'pink')
plt.show()