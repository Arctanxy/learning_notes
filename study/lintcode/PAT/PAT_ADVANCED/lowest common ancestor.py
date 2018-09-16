M,N = [int(t) for t in input().split(' ')]

nums = [int(t) for t in input().split(' ')]

for i in range(M):
    u,v = [int(t) for t in input().split(' ')]
    uin,vin = True,True
    if u not in nums:
        uin = False
    if v not in nums:
        vin = False
    if not uin and not vin:
        print("ERROR: %d and %d are not found." % (u,v))
    elif not uin and vin:
        print("ERROR: %d is not found." % u)
    elif not vin and uin:
        print("ERROR: %d is not found." % v)

    else:
        for n in nums:
            if u > v:
                if u > n and n > v:
                    print("LCA of %d and %d is %d." % (u,v,n))
                    break
                elif u == n:
                    print("%d is an ancestor of %d." % (u,v))
                    break
                elif v == n:
                    print("%d is an ancestor of %d." % (v,u))
                    break
            if u < v:
                if u < n and n < v:
                    print("LCA of %d and %d is %d." % (u,v,n))
                    break
                elif u == n:
                    print("%d is an ancestor of %d." % (u,v))
                    break
                elif v == n:
                    print("%d is an ancestor of %d." % (v,u))
                    break
            if u == v:
                print("%d is an ancestor of %d." % (u,v))
                break

'''
6 8
6 3 1 2 5 4 8 7
2 5
8 7
1 9
12 -3
0 8
99 99

'''