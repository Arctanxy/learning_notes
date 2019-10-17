def run():
    cmd = input().split(' ')
    A = int(cmd[0])
    B = int(cmd[1])
    R = A % B
    Q = int(A//B)
    print(str(Q) + ' ' + str(R))

if __name__ == "__main__":
    run()