def run():
    N = int(input())
    for i in range(N):
        num = input()
        num1 = int(num[:int(len(num)/2)])
        num2 = int(num[int(len(num)/2):])
        if num1 == 0 or num2 == 0:
            print('No')
            continue
        if int(num) % (num1 * num2) == 0:
            print('Yes')
        else:
            print("No")

run()

