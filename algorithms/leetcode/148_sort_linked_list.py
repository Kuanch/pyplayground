from utils.list_utils import list_to_linked_list
from utils.leetcode_object import ListNode


def sortList(head):
    if head is None or head.next is None:
        return head

    return split(head)


def split(node):
    pre = slow = fast = node
    while fast and fast.next:
        pre = slow
        slow = slow.next
        fast = fast.next.next
    pre.next = None
    if node.next:
        node = split(node)
    if slow.next:
        slow = split(slow)

    return merge(node, slow)


def merge(l1, l2):
    if l1 is None:
        return l2
    if l2 is None:
        return l1

    if l1.val < l2.val:
        l1.next = merge(l1.next, l2)
        return l1
    else:
        l2.next = merge(l1, l2.next)
        return l2


if __name__ == '__main__':
    assert sortList(list_to_linked_list([4, 2, 1, 3])) == list_to_linked_list([1, 2, 3, 4])
    assert sortList(list_to_linked_list([-1, 5, 3, 4, 0])) == list_to_linked_list([-1, 5, 3, 4, 0])
