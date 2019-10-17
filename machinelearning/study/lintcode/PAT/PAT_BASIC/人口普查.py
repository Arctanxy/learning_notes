def run():
    num = int(input())
    info = {}
    count = 0
    for i in range(num):
        name,birthday = input().split(' ')
        day = birthday
        today = '2014/09/06'
        pastday = '1814/09/06'
        if compare(day,today) or compare(pastday,day):
            continue
        else:
            info[name] = birthday
            count += 1
    # print(info)
    info = sorted(info.items(),key=lambda x: x[1])
    # print(info)
    print(count,info[0][0] ,info[-1][0])



def compare(day,pivot):
    # 判断day是否大于等于pivot
    day,pivot = [int(d) for d in day.split('/')],[int(p) for p in pivot.split('/')]
    for i in range(3):
        if day[i] > pivot[i]:
            return True
        elif day[i] < pivot[i]:
            return False
        else:
            continue
    return False
'''
5
John 2001/05/12
Tom 1814/09/06
Ann 2121/01/30
James 1814/09/05
Steve 1967/11/20
'''
if __name__ == "__main__":
    run()