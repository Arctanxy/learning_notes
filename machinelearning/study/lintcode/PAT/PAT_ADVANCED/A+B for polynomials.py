# input sample
# 2 1 2.4 0 3.2 ==> K = 2 polynomial:2.4x^1 + 3.2x^0
# 2 2 1.5 1 0.5 ==> K = 2 polynomial:1.5x^2 + 0.5x^1
# sum ==> 1.5x^2 + 2.9x + 3.2
# output ==> 3 2 1.5 1 2.9 0 3.2

case1 = input().split(' ')
case2 = input().split(' ')

# 获取多项式项数和最大次数
K1 = int(case1[0])
max_exp1 = int(case1[1])
K2 = int(case2[0])
max_exp2 = int(case2[1])
max_exp = max(max_exp1,max_exp2)
result = [0 for i in range((max_exp+1)*2)]
poly1 = [0 for i in range((max_exp+1)*2)]
poly2 = [0 for i in range((max_exp+1)*2)]
for i in range(1,len(case1)):
    # 奇数位为指数
    if i % 2 != 0:
        poly1[-(int(case1[i]) + 1)*2] = int(case1[i])
    # 偶数位为系数
    else:
        poly1[-(int(case1[i-1]) + 1)*2 +1] = round(float(case1[i]),1)

for i in range(1,len(case2)):
    # 奇数位为指数
    if i % 2 != 0:
        poly2[-(int(case2[i]) + 1)*2] = int(case2[i])
    # 偶数位为系数
    else:
        poly2[-(int(case2[i-1]) + 1)*2 +1] = round(float(case2[i]),1)


for i in range(len(poly1)):
    # 奇数位为系数
    if i % 2 != 0:
        result[i] = poly1[i] + poly2[i]
    else:
        result[i] = max(poly1[i],poly2[i])

output = ""

count = 0
for i in range(len(result)):
    # 奇数位为指数
    if i%2 == 0:
        if result[i + 1] != 0:
            output += ' ' + str(result[i])
            count += 1
        else:
            continue
    else:
        if result[i] != 0:
            output +=  ' '  + str(result[i])
        else:
            continue

print(str(count) + output)
