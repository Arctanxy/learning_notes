from collections import defaultdict
def run():
    A_info = input().split(' ')
    AN,A_dict = parser(A_info)
    B_info = input().split(' ')
    BN,B_dict = parser(B_info)

    max_length = max(A_dict.keys()) + max(B_dict.keys())+1

    res = {max_length-i-1:0 for i in range(max_length)}

    for k1,v1 in res.items():
        for k2,v2 in A_dict.items():
            if k2 <= k1:
                res[k1] += round(v2 * B_dict[k1-k2],1)

    result = ""
    count = 0
    for k,v in res.items():
        if v != 0:
            result += str(k) + ' ' + str(v) + ' '
            count += 1
        else:
            continue
    if count != 0:
        print(str(count) + ' ' + result[:-1])
    else:
        print(0)


def parser(info):
    N = info[0]
    coefs = info[1:]
    i = 0
    n_dict = defaultdict(int)
    while i <= len(coefs) - 2:
        n_dict[int(coefs[i])] = float(coefs[i+1])
        i += 2

    return N,n_dict

run()