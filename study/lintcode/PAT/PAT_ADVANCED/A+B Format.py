a,b = input().split(' ')
c = int(a) + int(b)
pivot = ""
if c >= 0:
    c = str(c) # 计算和
else:
    c = str(c)[1:]
    pivot = "-"
start = 0
end = 0
nums = []
for i in range(len(c)):
    if (len(c) - i) % 3 == 0:
        end = i
        if end != start:
            nums.append(c[start:end])
        start = end
nums.append(c[end:len(c)])
result = ','.join(nums)
print(pivot + result)