'''
冒泡排序

基本思路：

    1. 有长度为n的数组array，比较array[0]和array[1]，如果array[0]>array[1]则交换array[0]和array[1];
    2. 继续往后遍历array[i],array[i+1]，如果array[i]>array[i+1]则交换array[i]和array[i+1];
    3. 如果交换之后array[i-1]>array[i]，则继续往前交换；
    4. 直到array[k-1]<array[k]为止。

'''

def bubble_sort(lst):

    for i in range(len(lst)-1):
        for j in range(i+1,0,-1):
            if lst[j]<lst[j-1]:
                lst[j],lst[j-1] = lst[j-1],lst[j]
                #j -= 1
    print(lst)

if __name__ == "__main__":
    lst = [1,2,42,1,3,4,6,2,5,7,784,23]
    bubble_sort(lst)