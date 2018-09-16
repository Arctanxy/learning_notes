# 连接的权重为两人通话的总时长
# 通话权重大于阈值K的2人以上团体为GANG
# 一个GANG中通话权重最大的是头目
from collections import defaultdict

N,K = [int(t) for t in input().split(' ')]
name_list = []
# 记录每个人的通话时间
time_dict = defaultdict(int)
# 记录每个人的GANG编号
gang_dict = {}
g = 0
for i in range(N):
    info = input().split(' ')
    name1,name2,p_time = info[0],info[1],int(info[2])
    if name1 not in name_list:
        name_list.append(name1)
    if name2 not in name_list:
        name_list.append(name2)
    time_dict[name1] += p_time
    time_dict[name2] += p_time
    if name1 in gang_dict and name2 not in gang_dict:
        gang_dict[name2] = gang_dict[name1]
    elif name2 in gang_dict and name1 not in gang_dict:
        gang_dict[name1] = gang_dict[name2]
    elif name1 not in gang_dict and name2 not in gang_dict:
        g += 1
        gang_dict[name1] = g
        gang_dict[name2] = g
    elif name1 in gang_dict and name2 in gang_dict:
        for k,v in gang_dict.items():
            if v == gang_dict[name1]:
                gang_dict[k] = gang_dict[name2]


g = 0
gangs = []
while g <= max(gang_dict.values()):
    gang = {k:v for k,v in gang_dict.items() if v == g}
    times = {k:v for k,v in time_dict.items() if k in gang}
    if len(gang) > 2 and sum(times.values())/2 > K:
        gangs.append(gang)
    g += 1
print(len(gangs))
for gang in gangs:
    time_g = {k:v for k,v in time_dict.items() if k in gang}
    time_g = sorted(time_g.items(),key=lambda x:x[1],reverse=False)
    info = time_g.pop()
    print(str(info[0]) + ' ' + str(len(gang)))




