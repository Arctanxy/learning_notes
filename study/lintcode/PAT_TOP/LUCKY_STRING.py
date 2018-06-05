def fibonacci(m):
    '''
    递归法求斐波那契数列，但是不适合于本算法
    :param m:
    :return:
    '''
    if m == 0:
        return m
    elif m == 1:
        return m
    else:
        return fibonacci(m-1)+fibonacci(m-2)

s = input("input:")

def get_fibonacci():
    nums = [0,1]
    i = 2
    new_item = 0
    while new_item <= 26:
        new_item = nums[i-1] + nums[i-2]
        if new_item > 26:
            break
        nums.append(new_item)
        i += 1
    return list(set(nums))

def lucky_or_not(s,f):
    luck_list = []
    for i in range(len(s)):
        for j in range(i+1,len(s)+1):
            sub_s = s[i:j]
            if len(set(sub_s)) in f:
                luck_list.append(sub_s)
    return luck_list

f = get_fibonacci()
ls = lucky_or_not(s,f)
ls = list(set(ls))
print(sorted(ls))
