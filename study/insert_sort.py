'''
插入排序

基本思路：
    1. 从第一个元素开始向后遍历；
    2. 对于任意元素array[i]，向前寻找array[j]，若array[j]>array[i]则继续向前寻找，并将array[j]对应的元素后移一位；
    3. 若array[j]<=array[i]，则将array[i]插入到array[j+1]的位置；
    4. 对i = 2,3...n依次重复2——3操作。
'''
import matplotlib.pyplot as plt
import time

def insert_sort(lst):
    lsts = []
    for i in range(len(lst)):
        temp = lst[i]
        j = i-1
        while j>=0 and lst[j]>temp:
            lst[j+1] = lst[j]
            j -= 1
        lst[j+1] = temp
        l = lst[:]
        lsts.append(l)
    return lsts


if __name__ == "__main__":
    lst = [13,32,42,1,53,4,66,2,5,7,74,23]
    lsts = insert_sort(lst)
    plt.ion()
    fig = plt.figure()
    ax  = plt.gca()
    bars = ax.bar(range(len(lst)),height=lst)
    for l in lsts:
        print(l)
        bars.remove()
        bars = ax.bar(range(len(lst)),height=l)
        plt.pause(0.5)
    while True:#防止图片关闭
        plt.pause(1)
    
