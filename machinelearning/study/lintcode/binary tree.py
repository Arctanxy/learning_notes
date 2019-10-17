# 前序遍历
# 根节点->左子树->右子树

# 按前序创建一个二叉树

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def create_tree(root, lst):
    global i
    # i不能超出lst范围，并且遇到#意味着到了叶节点，无法继续
    if i > len(lst) -1 or lst[i] == '#':
        return None
    else:
        data = lst[i]
        root = TreeNode(data)
        i += 1
        root.left = create_tree(root.left, lst)
        i += 1
        root.right = create_tree(root.right, lst)
    return root


def traversal(root,itype='preorder'):
    # 递归遍历主函数
    res = []
    if itype == 'preorder':
        preorder(root,res)
    elif itype == 'inorder':
        inorder(root, res)
    elif itype == 'postorder':
        postorder(root,res)
    else:
        print("Wrong itype.")
    print(res)


def inorder(root, res):
    # 中序遍历
    if root is None:
        return None
    else:
        inorder(root.left, res)
        res.append(root.val)
        inorder(root.right, res)


def inorder_iter(root):
    res, stack = [], []
    node = root
    if root is None:
        return None

    while node or len(stack) > 0:
        if node:
            # 一直向左，添加所有的左子树值
            stack.append(node)
            node = node.left
        else:
            # 逐步回退
            node = stack.pop()
            res.append(node.val)
            node = node.right
    return res


def preorder(root, res):
    # 前序遍历
    if root is None:
        return None
    else:
        res.append(root.val)
        preorder(root.left,res)
        preorder(root.right,res)


def preorder_iter(root):
    res, stack = [], []
    if root is None:
        return None
    print("结点%s,入栈" % root.val)
    stack.append(root)
    while len(stack) > 0:
        node = stack.pop()
        print("结点%s,出栈入表" % node.val)
        res.append(node.val)
        if node.right:
            stack.append(node.right)
            print("发现结点%s存在右结点，右结点%s入栈" % (node.val,node.right.val))
        if node.left:
            stack.append(node.left)
            print("发现结点%s存在左结点，左结点%s入栈" % (node.val, node.left.val))
    return res


def postorder(root, res):
    # 后序遍历
    if root is None:
        return None
    else:
        postorder(root.left,res)
        postorder(root.right,res)
        res.append(root.val)


def postorder_iter(root):
    # 后序迭代遍历
    res, stack1, stack2 = [], [], []
    if root is None:
        return None
    stack1.append(root)
    while len(stack1) > 0:
        node = stack1.pop()
        stack2.append(node)
        if node.left:
            stack1.append(node.left)
        if node.right:
            stack1.append(node.right)

    while len(stack2) > 0:
        node = stack2.pop()
        res.append(node.val)
    return res


if __name__ == "__main__":
    lst = ['1','2','4','#','#','5','#','#','3','#','6','#']
    root = TreeNode(0)
    i = 0
    root = create_tree(root, lst)
    res = postorder_iter(root)
    print(res)