def run():
    command = [int(i) for i in input().split(' ')]
    result = []
    # 筛选出第一个数字
    first_num = min([x for x in command if x > 0])
    result.append(first_num)
    # 去除除0外的最小数
    command.remove(first_num)
    # 剩余数字进行排序
    command = sorted(command)
    result.extend(command)
    print(''.join([str(i) for i in result]))


if __name__ == "__main__":
    run()