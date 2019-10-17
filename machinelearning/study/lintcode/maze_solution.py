# 使用栈求解迷宫问题


maze = [
    [1,0,0,0,1],
    [1,1,1,0,1],
    [1,0,1,1,1],
    [0,1,1,0,1]
]

start = (0,0)
end = (3,3)

def find_first_direction(maze,pos):
    directions = [
        [pos[0] + 1,pos[1]]
    ]

    for dir in directions:
        if maze[dir[0]][dir[1]] == 1:
            maze[dir[0]][dir[1]] = 2 # 访问过的节点置为2
            return maze[dir[0]][dir[1]]

def get_direction(maze,pos):
    # 返回第一个可行的方向
    # 如果是左上角点
    maze[pos[0]][pos[1]] = 2
    if pos[0] == 0 and pos[1] == 0:
        directions = [
            [pos[0] + 1,pos[1]], # 下边
            [pos[0],pos[1] + 1] # 右边
        ]
        return find_first_direction(maze,directions)
    # 如果在第一行
    elif pos[0] == 0 and pos[1] != 0:
        directions = [
            [pos[0] + 1,pos[1]],
            [pos[0],pos[1] + 1],
            [pos[0],pos[1] - 1] # 左边
        ]

