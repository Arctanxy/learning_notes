s1 = input()
s2 = input()

result = []
for c in s1:
    if c not in s2:
        result.append(c)
result = ''.join(result)

print(''.join(result))
