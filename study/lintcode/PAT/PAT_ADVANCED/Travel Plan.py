'''
A traveler's map gives the distances between cities along the highways, together with the cost of each highway.
Now you are supposed to write a program to help a traveler to decide the shortest path between his/her starting
city and the destination. If such a shortest path is not unique, you are supposed to output the one with the minimum
 cost, which is guaranteed to be unique.

Input Specification:
Each input file contains one test case. Each case starts with a line containing 4 positive integers N, M, S, and D,
 where N (≤500) is the number of cities (and hence the cities are numbered from 0 to N−1); M is the number of highways;
 S and D are the starting and the destination cities, respectively. Then M lines follow, each provides the information
  of a highway, in the format:

City1 City2 Distance Cost
where the numbers are all integers no more than 500, and are separated by a space.

Output Specification:
For each test case, print in one line the cities along the shortest path from the starting point to the destination,
followed by the total distance and the total cost of the path. The numbers must be separated by a space and there must
 be no extra space at the end of output.

Sample Input:
4 5 0 3
0 1 1 20
1 3 2 30
0 3 4 10
0 2 2 20
2 3 1 20


4 5 0 1
0 1 1 20
1 3 2 30
0 3 4 10
0 2 2 20
2 3 1 20
Sample Output:
0 2 3 3 40
'''

def printf(lst,label = ""):
    for r in lst:
        print(r)

N,M,S,D = [int(t) for t in input().split(' ')]

inf = 99999999
distance = [[inf for i in range(N)] for i in range(N)]
cost = [[inf for i in range(N)] for i in range(N)]
for i in range(M):
    info = [int(t) for t in input().split(' ')]
    city1,city2 = info[0:2]
    distance[city1][city2] = info[2]
    cost[city1][city2] = info[3]


# 提取出到出发城市的距离
dis = distance[S]
cos = cost[S]

def Dijkstra():
    global distance
    global dis

    # u = [{current_city:[pre_city,distance,cost]}]
    U = [[S,[0],0,0]]
    cities = [i for i in range(N)]
    cities.remove(S)
    # v = [{current_city:[pre_city,distance,cost]}]
    V = [[city,[0],dis[city],cos[city]] for city in cities]
    # item = U[0]
    while V:
        V = sorted(V,key=lambda x:(x[1],x[2]),reverse=True)
        # 提取出最后一个元素，即distance和cost最小的元素
        item = V.pop()

        # 加入到U中
        U.append(item)
        # 提前结束条件

        if item[0] == D:
            break
        for i,c in enumerate(V):
            # 如果distance(0,c) > distance(k,c) + distance(0,k)
            # 则更新c的distance，并将c的前置点设为k
            if dis[c[0]] > distance[item[0]][c[0]] + dis[item[0]]:
                # 更新前置点
                V[i][1] = [item[0]]
                # 更新距离
                dis[c[0]] = distance[item[0]][c[0]] + dis[item[0]]
                distance[S][c[0]] = distance[item[0]][c[0]] + dis[item[0]]
            elif dis[c[0]] == distance[item[0]][c[0]] + dis[item[0]]:
                # 添加前置点
                V[i][1].append(item[0])
                # 距离不更新
            else:
                pass

    return U

visited = {c:False for c in range(N)}
path = []
mincos = inf
best_path = []
def dfs(s,ind,d,U,visited,path):
    # global paths
    global mincos
    global best_path
    path.append(ind)
    print(path)
    if ind == d:
        print('=',path)
        current_cos = 0
        for i in range(len(path)-1):
            current_cos += cost[path[i+1]][path[i]]
        if current_cos <= mincos:
            mincos = current_cos
            best_path = path.copy()
        path.pop()
        print(path)
    else:
        print('e',path)
        for i in range(N):
            # print('i',i,'ind',ind,i in U[ind][0] and visited[i] == False)
            if i in U[ind][0]:
                dfs(s,i,d,U,visited,path)
        path.pop()
        print(path)

U = Dijkstra()
U = {k[0]:k[1:4] for k in U}
dfs(D,D,S,U,visited,path)
total_distance = 0
for i in range(len(best_path)-1):
    total_distance += distance[best_path[i+1]][best_path[i]]

print(' '.join([str(p) for p in best_path[::-1]]) + ' ' + str(total_distance) + ' ' + str(mincos))
