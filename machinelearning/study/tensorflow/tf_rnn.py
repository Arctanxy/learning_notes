
#加载ptb数据
#reader.py文件需要自行从https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/reader.py下载
import reader

DATA_PATH = r"C:\Users\Dl\Documents\GitHub\learning_notes\data\simple-examples\data"
train_data,valid_data,test_data,_ = reader.ptb_raw_data(DATA_PATH)

x,y = reader.ptb_producer(train_data,4,5)
print(x)

print(y)
