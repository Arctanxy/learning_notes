'''
1004 Counting Leaves (30)（30 分）
A family hierarchy is usually presented by a pedigree tree. Your job is to count those family members who have no child.

Input

Each input file contains one test case. Each case starts with a line containing 0 < N < 100, the number of nodes in a tree,
 and M (< N), the number of non-leaf nodes. Then M lines follow, each in the format:

ID K ID[1] ID[2] ... ID[K]
where ID is a two-digit number representing a given non-leaf node, K is the number of its children, followed by a sequence
 of two-digit ID's of its children. For the sake of simplicity, let us fix the root ID to be 01.

Output

For each test case, you are supposed to count those family members who have no child for every seniority level starting
from the root. The numbers must be printed in a line, separated by a space, and there must be no extra space at the end of
 each line.

The sample case represents a tree with only 2 nodes, where 01 is the root and 02 is its only child. Hence on the root 01
level, there is 0 leaf node; and on the next level, there is 1 leaf node. Then we should output "0 1" in a line.

Sample Input

4 2
01 1 02
02 2 03 04
Sample Output

0 1
'''

from collections import defaultdict

N,M = [int(t) for t in input().split(' ')]

mat = {}
for i in range(M):
    info = [int(t) for t in input().split(' ')]
    mat[info[0]] = info[1:]

result = defaultdict(int)
level = 0

def count(mat,p,level):
    if p not in mat:
        result[level] += 1
    else:
        for i in range(mat[p][0]):
            count(mat,mat[p][i+1],level+1)

count(mat,1,0)


r_list = ['0' for i in range(max(result.keys()) + 1)]
for k,v in result.items():
    r_list[k] = str(v)
print(' '.join(r_list))