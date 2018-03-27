'''
快速排序

基本思路：

    1. 找一个基准值temp，一般选择数组的最后一个元素值；
    2. 将大于temp的元素都放在temp后面，将小于temp的元素都放在temp前面；
    3. 得到大于temp和小于等于temp两个数组；
    4. 对这两个数组递归调用分组函数
'''


def partition(lst,low,high):
    '''
    low是起始index
    high是终止index
    lst是数组
    '''
    temp = lst[high-1]
    small = low-1
    for i in range(low,high):
        if lst[i]<temp:
            small += 1#small用于记录最后一个小于temp的元素的index
            if small != i:
                lst[small],lst[i] = lst[i],lst[small]#交换的出发条件是发现了比temp小的元素
    if lst[small+1] > temp:
        lst[small+1],lst[high-1] = lst[high-1],lst[small+1]
    return small+1

def quick_sort(lst,low,high):
    if low < high:
        p = partition(lst,low,high)
        quick_sort(lst,low,p)
        quick_sort(lst,p+1,high)
    return None



if __name__ == "__main__":
    lst = [13,32,42,1,53,4,66,2,5,7,74,23]
    quick_sort(lst,0,len(lst))
    print(lst)

