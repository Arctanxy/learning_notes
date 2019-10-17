class Solution:
    def maxIncreaseKeepingSkyline(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        row_max = []
        for i in range(len(grid)):
            row_max.append(max(grid[i]))
        col_max = []
        for j in range(len(grid[0])):
            col_max.append(max([g[j] for g in grid]))


        _sum = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                _sum += min(row_max[i],col_max[j]) - grid[i][j]
        return _sum

if __name__ == "__main__":
    s = Solution()
    grid = [[3,0,8,4],[2,4,5,7],[9,2,6,3],[0,3,1,0]]
    result = s.maxIncreaseKeepingSkyline(grid)
    print(result)
