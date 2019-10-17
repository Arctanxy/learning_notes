def run():
    _input = input().split(' ')
    num = int(_input[0])
    char = _input[1]
    '''
    # 计算得到组成沙漏的元素的通项公式为an = 2*n^2 - 1，n为半边沙漏的层数
    # 1000以内组成沙漏的最大n为22
    left = 0
    right = 22
    n = 11
    while left < right:
        if cal(n) > num:
            right = n
        elif cal(n) <= num:
            if cal(n + 1) > num:
                # print('got n: %d ' % n)
                break
            else:
                left = n
        else:
            pass
        n = int((left+right)/2)
    result = num - cal(n)

    for i in range(n):
        print(' ' * i + char * (2*(n-i)-1) + ' '*i)

    for j in range(1,n):
        print(' ' * (n-j-1) + char * (2*(j+1) -1) + ' '*(n-j-1))

    print(result)

    '''

    n = 1
    while cal(n) < num:
        n += 1
    if n > 1:
        n -= 1
    # print(n)

    result = num - cal(n)
    # print(num,cal(n))

    for i in range(n):
        print(' ' * i + char * (2 * (n - i) - 1) + ' ' * i)

    for j in range(1, n):
        print(' ' * (n - j - 1) + char * (2 * (j + 1) - 1) + ' ' * (n - j - 1))


    print(result)

def cal(n):
    return 2*n**2 -1



if __name__ == "__main__":
    run()