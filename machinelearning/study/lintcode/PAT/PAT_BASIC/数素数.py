import math
def run():
    p = input().split(' ')
    pm = int(p[0])
    pn = int(p[1])
    # 数字，素数从2开始
    # pm = 5
    # pn = 27
    n = 2
    # 序号
    i = 0
    # 素数组
    primes = []
    while i < pn:
        prime = True
        if n < 4:
            pass
        else:
            for k in range(2,int(math.sqrt(n))+1):
                if n % k == 0:
                    prime = False
                    break
        if prime:
            primes.append(n)
            i += 1
        n += 1

    for j,p in enumerate(primes[pm-1:]):
        if (j+1) % 10 == 0:
            print(p)
        elif j == len(primes[pm-1:]) - 1:
            print(p)
        else:
            print(p,end=' ')

if __name__ == "__main__":
    run()