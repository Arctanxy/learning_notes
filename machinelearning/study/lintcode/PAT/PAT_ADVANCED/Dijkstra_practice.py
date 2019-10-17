graph = [
[2,3],
[4],
[4,6],
[5],
[6],
]
inf = 99999999
distance = [
    [0,1,2,inf,inf,inf],
    [inf,0,inf,1,inf,inf],
    [inf,inf,0,2,inf,2],
    [inf,inf,inf,0,1,inf],
    [inf,inf,inf,inf,0,1],
    [inf,inf,inf,inf,inf,0]
]


S = 1
D = 6

def Dijkstra(graph):
    dis = distance[S-1]
    # 最初U中只包含起点
    U = [[S,[None],0]]
    points = [i+1 for i in range(6)]
    points.remove(S)
    # V中包含除起点外的其他店
    V = [[c,[S],dis[c-1]] for c in points]
    while V:
        # 从大到小排列
        V = sorted(V,key=lambda x:x[2],reverse=True)
        # 从V中取距离S最近的点，加入U中
        item = V.pop()
        U.append(item)
        # 如果找到了终点就提前停止
        if item == D:
            break
        # 遍历V，更新各点距离
        for i,c in enumerate(V):
            # 如果某点到S的距离小于某点经由item[0]（上一个找到的点）到S的距离
            # 则更新该点坐标，并将上一个找到的点作为该点的前置点
            if dis[c[0]-1] > dis[item[0]-1] + distance[item[0]-1][c[0]-1]:
                dis[c[0]-1] = dis[item[0]-1] + distance[item[0]-1][c[0]-1]
                distance[S-1][c[0]-1] = dis[item[0]-1] + distance[item[0]-1][c[0]-1]
                # 更新前置点
                V[i][1] = [item[0]]
                # 更新距离
                V[i][2] = dis[c[0]-1]
            elif dis[c[0]-1] == dis[item[0]-1] + distance[item[0]-1][c[0]-1]:
                # 添加前置点
                V[i][1].append(item[0])
            else:
                pass
    return U

U = Dijkstra(graph)


print(U)
