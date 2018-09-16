start,N,K = [t for t in input().split(' ')]
N = int(N)
K = int(K)
data_dict = {}
for i in range(N):
    info = input().split(' ')
    data_dict[info[0]] = (info[1],info[2])

class Linked_list:
    def __init__(self,x,address,next_address):
        self.val = x
        self.address = address
        self.next_address = next_address
        self.next = None
        self.parent = None

def create_linked_list(data_dict,start,parent):
    if start not in data_dict:
        return None
    root = Linked_list(
        int(data_dict[start][0]),
        start,
        data_dict[start][1],
    )
    root.parent = parent

    root.next = create_linked_list(
        data_dict,
        root.next_address,
        root
    )

    return root
root = create_linked_list(data_dict,start,None)
# 建表成功
print(root.val,root.next.val)

def swap_nodes(A,B):
    B.parent = A.parent
    A.next = B.next
    A.parent = B
    B.next = A
    return B,A

node = root
lst = [node]
while node.next:
    lst.append(node.next)
    node = node.next

print([l.val for l in lst])

negative = []
k_g = []
for item in lst:
    if item.val < 0:
        k_g


