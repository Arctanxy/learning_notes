# 第一对相同的大写英文字母代表星期几
# 第二对相同的英文字母代表小时
# 第二个字符串中第一对相同的英文字母出现的位置代表分钟。


def run():
    week = {'A':'MON','B':'TUE',
            'C':'WED','D':'THU',
           'E':'FRI','F':'SAT','G':'SUN'}
    hour = {
        '0':0,'1':1,'2':2,'3':3,'4':4,'5':5,
        '6':6,'7':7,'8':8,'9':9,'A':10,'B':11,
        'C':12,'D':13,'E':14,'F':15,'G':16,'H':17,'I':18,'J':19,
        'K':20,'L':21,'M':22,'N':23
    }
    a,b,c,d = input(),input(),input(),input()
    w = None
    h = None
    if len(a) <= len(b):
        sample1 = a

    else:
        sample1 = b
    for i in range(len(sample1)):
        # 寻找第一对大写英文字母
        if a[i].isalpha() and a[i] == b[i] and a[i].isupper() and w is None:
            w = week[a[i]]
            continue
        if w is not None and a[i] == b[i] and (a[i].isalpha() or a[i].isdigit()):
            h = hour[a[i].upper()]
            break

    if len(c) <= len(d):
        sample2 = c
    else:
        sample2 = d
    for j in range(len(sample2)):
        if c[j] == d[j] and c[j].isalpha():
            minute = j
            break

    if h<10:
        h = str(0) + str(h)
    if minute<10:
        minute = str(0) + str(minute)
    print(w + ' ' + str(h) + ':' + str(minute))




if __name__ == "__main__":
    run()