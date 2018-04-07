import urllib.request
import json
import pandas as pd 
from tqdm import tqdm
import threading
import time

FOLDER = 'H:/learning_notes/project/kaggle/iMaterialist Challenge/'


def download_data(is_type = 'train'):
    time.sleep(0.5)
    images = pd.read_json(FOLDER + is_type + '.json')
    images['url'] = images['images'].apply(lambda x:x['url'][0])
    images['name'] = images['url'].apply(lambda x:x.split('/')[-1])
    images['id'] = images['images'].apply(lambda x:x['image_id'])
    del images['images']
    if is_type == 'train':
        images['label'] = images['annotations'].apply(lambda x:x['label_id'])#图片分类标签
        del images['annotations']
        #images.to_csv(FOLDER + is_type + '_data.csv',index = False)#已经保存过了
        print('total num:%d' % images.shape[0])
        for i,row in images[8000:].iterrows():
            if i%10 == 0:
                print('train %d' % i)
            try:
                urllib.request.urlretrieve(row['url'],filename=FOLDER + is_type + '_image/' + row['name'])
            except:
                pass
    elif is_type == 'validation':
        images['label'] = images['annotations'].apply(lambda x:x['label_id'])#图片分类标签
        del images['annotations']
        #images.to_csv(FOLDER + is_type + '_data.csv',index = False)#已经保存过了
        print('total num:%d' % images.shape[0])
        for i,row in images[5000:].iterrows():
            if i%10 == 0:
                print('test %d' % i)
            try:
                urllib.request.urlretrieve(row['url'],filename=FOLDER + is_type + '_image/' + row['name'])
            except:
                pass
    else:
        print('total num:%d' % images.shape[0])
        for i,row in images[3000:].iterrows():
            if i%10 == 0:
                print('test %d' % i)
            try:
                urllib.request.urlretrieve(row['url'],filename=FOLDER + is_type + '_image/' + row['name'])
            except:
                pass
    
if __name__ == "__main__":
    threads = []
    for item in ['test','train','validation']:
        threads.append(threading.Thread(target=download_data,args=(item,)))
    for t in threads:
        t.start()
        #t.join()

    '''
    2018年4月6日21:33:20
    test:537个文件
    train:3884个文件
    validation：1454个文件
    '''