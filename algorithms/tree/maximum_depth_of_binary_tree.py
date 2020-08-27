from tree.tree_utils import TreeNode
from tree.tree_utils import TestCase1


class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if root is None:
            return 0

        return self.visit(root, 1)

    def visit(self, node, depth):
        l_depth = r_depth = depth
        if node.left is not None:
            l_depth = self.visit(node.left, l_depth + 1)
        if node.right is not None:
            r_depth = self.visit(node.right, r_depth + 1)
        return max(l_depth, r_depth)


def main():
    assert 3 == Solution().maxDepth(TestCase1.root)


if __name__ == '__main__':
    main()
