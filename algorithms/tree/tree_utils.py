class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class TestCase1(object):
    leaf1 = TreeNode(15)
    leaf2 = TreeNode(7)
    node1 = TreeNode(20, leaf1, leaf2)
    node2 = TreeNode(9)
    root = TreeNode(3, node2, node1)
