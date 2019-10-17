class TreeNode:
    def __init__(self,x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def averageOfLevels(self,root):
        def dfs(node,depth = 0):
            info = []
            if node:
                if len(info) <= depth:
                    info.append([0,0])
                info[depth][0] += node.val
                info[depth][1] += 1.0
                dfs[node.left,depth+1]
                dfs[node.right,depth+1]
        dfs(root)
        return [s[0]/s[1] for s in info]
