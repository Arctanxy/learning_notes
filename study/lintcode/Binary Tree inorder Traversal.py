class TreeNode:
    def __init__(self,x):
        self.val = x
        self.left = None
        self.right = None


class Solution:

    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        self.result = []
        return self.result

    def inorder(self,root):
        if root is None:
            return None

        self.result.append(self.inorder(root.left))
        self.result.append(self.inorder(root.right))
        self.result.append(root.val)


