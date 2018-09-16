def mark(mat):
    # 四领域计算连通域
    marker = [[0 for i in range(len(mat[0]))] for j in range(len(mat))]
    # 连通域标签
    label = 0
    for m in marker:
        print(m)
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            # 如果当前点是0，直接跳过
            if mat[i][j] == 0:
                continue
            # 如果是1，分情况讨论
            else:
                # 左上起点
                if i == j and j == 0:
                    label += 1
                    marker[i][j] = label
                # 第一行
                elif i == 0:
                    if mat[i][j-1] == 1:
                        marker[i][j] = marker[i][j-1]
                    else:
                        label += 1
                        marker[i][j] = label
                # 第一列
                elif j == 0:
                    if mat[i-1][j] == 1:
                        marker[i][j] = marker[i-1][j]
                    else:
                        label += 1
                        marker[i][j] = label
                # i>0,j>0时
                else:
                    # 如果左边是1，上边是0
                    if mat[i][j-1] == 1 and mat[i-1][j] == 0:
                        marker[i][j] = marker[i][j-1]
                    elif mat[i][j-1] == 0 and mat[i-1][j] == 1:
                        marker[i][j] = marker[i-1][j]
                    # 如果两边都是小的，将三个格子都取小值
                    elif mat[i][j-1] == 1 and mat[i-1][j] == 1:
                        if marker[i][j-1] <= marker[i-1][j]:
                            marker[i][j] = marker[i][j-1]
                            marker[i - 1][j] = marker[i][j-1]
                        else:
                            marker[i][j] = marker[i-1][j]
                            marker[i][j-1] = marker[i-1][j]
                    # 两边都是0
                    else:
                        label += 1
                        marker[i][j] = label

    for n in marker:
        print(n)



if __name__ == "__main__":
    mat = [
        [1,1,1,0],
        [1,0,1,0],
        [0,1,0,1],
        [1,0,1,1],
        [1,1,1,0]
    ]
    mark(mat)