def run():
    n = int(input())
    i = 1
    while n > 0:
        case = input()
        nums = case.split(' ')
        nums = [int(n) for n in nums]
        if nums[0] + nums[1] > nums[2]:
            print('Case #%d: true' % i)
        else:
            print('Case #%d: false' % i)
        i += 1
        n -= 1
if __name__ == "__main__":
    run()
