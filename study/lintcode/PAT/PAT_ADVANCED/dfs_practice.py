graph = [
[2,3],
[4],
[4,6],
[5],
[6],
]

paths = []
path = []
def dfs(start,index,end,graph):
    path.append(index)
    if index == end:
        paths.append(path.copy())
        path.pop()
        print("找到终点%d，得到路径，往前回溯一位，查看节点%d是否有其他路径" % (index,path[-1]))

    else:
        print("依次搜索节点%d，%d的后置节点有 %s"% (index,index,str(graph[index-1])))
        for item in graph[index-1]:
            print("搜索节点%d的后置节点%d" % (index,item))
            if item not in path:
                dfs(start,item,end,graph)
        path.pop()
        if path != []:
            print("节点%d的后置节点搜索完毕，往前回溯一位，查看节点%d处是否有其他路径" % (index,path[-1]))
        else:
            print("循环结束，已无其他路径！")
dfs(1,1,6,graph)
for i,p in enumerate(paths):
    print("path %d" % i + str(p))