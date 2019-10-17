def check():
    M, N, K = [int(t) for t in input().split(' ')]
    for i in range(K):
        num = [int(t) for t in input().split(' ')]
        stack1 = []
        stack2 = [N - i for i in range(N)]
        possible = True
        for n in num:
            node = 0
            if n not in stack1:
                while node != n:
                    node = stack2.pop()
                    stack1.append(node)
                    if len(stack1) > M:
                        possible = False
                    # print('stack1',stack1,'stack2',stack2)
                if len(stack1) > M:
                    possible = False
                stack1.pop()
            if len(stack1) > M:
                possible = False
            if n in stack1:
                if n == stack1[-1]:
                    stack1.pop()
                else:
                    possible = False
        if possible:
            print("YES")
        else:
            print("NO")



check()