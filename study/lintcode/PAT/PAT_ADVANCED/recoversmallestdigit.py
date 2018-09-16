_input = input().split(' ')
N = _input[0]
nums = _input[1:]
def sort(nums):
    if len(nums) <= 1:
        return nums
    base = nums.pop(0)
    left = []
    right = []
    for n in nums:
        if int(n[0]) < int(base[0]):
            left.append(n)
        elif int(n[0]) > int(base[0]):
            right.append(n)
        # 若首数字相等
        else:
            min_len = min(len(n),len(base))
            got = False
            for i in range(1,min_len):
                if int(n[i]) < int(base[i]):
                    left.append(n)
                    got = True
                    break
                elif int(n[i]) > int(base[i]):
                    right.append(n)
                    got = True
                    break
                else:
                    if len(n) > len(base):
                        # 如果n中的m+1为比n的首字大，则说明在这个数后面再接别的数字不如把这个数接在相似数的后面好
                        if int(n[min_len]) > int(n[0]):
                            right.append(n)
                            break
                        elif int(n[min_len]) <= int(n[0]):
                            left.append(n)
                            break
                    elif len(n) < len(base):
                        if int(base[min_len]) >= int(base[0]):
                            left.append(n)
                            break
                        elif int(base[min_len]) < int(base[0]):
                            right.append(n)
                            break
    return sort(left) + [base] + sort(right)

if __name__ == "__main__":
    new = sort(nums)
    new_num = ''.join(new)
    print(int(new_num))