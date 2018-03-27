'''
选择排序

基本思路：
    1. 假设有数组array，长度为n，找到array中最小的元素，与array[0]进行交换；
    2. 找到array[1:]中最小的元素，与array[1]交换；
    3. 以此类推，最终得到有序数组
'''

def select_sort(lst):
    for i in range(len(lst)):
        min_index = i
        for j in range(i,len(lst)):
            if lst[j]<lst[min_index]:
                min_index = j
        lst[i],lst[min_index] = lst[min_index],lst[i]
    print(lst)
    



if __name__ == "__main__":
    lst = [13,32,42,1,53,4,66,2,5,7,74,23]
    select_sort(lst)