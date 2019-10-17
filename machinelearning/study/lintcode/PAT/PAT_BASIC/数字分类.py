def run():
    command = input().split(' ')
    count = command[0]
    nums = command[1:]
    a1,a2,a3,a4,a5 = [],[],[],[],[]
    for n in nums:
        n = float(n)
        if n % 10 == 0:
            a1.append(n)
        elif n % 5 == 1:
            a2.append(n)
        elif n % 5 == 2:
            a3.append(n)
        elif n % 5 == 3:
            a4.append(n)
        elif n % 5 == 4:
            a5.append(n)
    if len(a1) == 0:
        A1 = "N"
    else:
        A1 = int(sum(a1))
    if len(a2) == 0:
        A2 = "N"
    else:
        A2 = int(sum([n * pow((-1),i) for i,n in enumerate(a2)]))
    A3 = len(a3)
    if A3 == 0:
        A3 = "N"
    try:
        A4 = round(sum(a4)/len(a4),1)
    except:
        A4 = "N"
    if len(a5) == 0:
        A5 = "N"
    else:
        A5 = int(max(a5))
    print("%s %s %s %s %s" % (str(A1),str(A2),str(A3),str(A4),str(A5)))

if __name__ == "__main__":
    run()
