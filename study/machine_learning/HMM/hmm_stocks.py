import datetime
import numpy as np
import matplotlib.pyplot as plt
import tushare as ts
from hmmlearn.hmm import GaussianHMM
'''
http://hmmlearn.readthedocs.io/en/latest/auto_examples/plot_hmm_stock_analysis.html#sphx-glr-auto-examples-plot-hmm-stock-analysis-py
本段代码用于分析股票数据序列背后的隐藏序列，即解释股票状态
根据序列数据训练出模型之后，可以得到隐藏态以及转移矩阵和隐藏序列到观察序列的转移概率，便可以通过隐藏序列预测观察序列的变化。
'''
data = ts.get_hist_data('600848',start='2010-01-01',end='2017-12-31')

print(data.info())
close_v = data['close'].values
dates = np.array([i for i in range(data.shape[0])])
volume = data['volume'].values

diff = np.diff(close_v)#要训练的是收盘价格的变化值
dates = dates[1:]
close_v = close_v[1:]
volume = volume[1:]

plt.plot(close_v,color = 'blue')
plt.show()
plt.savefig("H:/learning_notes/study/machine_learning/HMM/stocks.jpg")

X = np.column_stack([diff,volume])
print(X)
diff = diff.reshape(-1,1)
model = GaussianHMM(n_components=2,n_iter=1000)#covariance_type="diag"为默认参数
model.fit(diff)

hidden_states = model.predict(diff)
print(hidden_states)
print(model.covars_)

for i in range(model.n_components):
    print("{0}th hidden state".format(i))
    print("mean=",model.means_[i])
    print("var=",np.diag(model.covars_[i]))

colors = ['r','g','b','y']

for j in range(len(close_v)-1):
    for i in range(model.n_components):
        if hidden_states[j] == i:
            plt.plot([dates[j],dates[j+1]],[close_v[j],close_v[j+1]],color = colors[i])

plt.show()
plt.savefig("H:/learning_notes/study/machine_learning/HMM/hidden_states.jpg")