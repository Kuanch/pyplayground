from utils.node_utils import ListNode, list_to_link


# Stack, T = O(n), S = O(n)
def isPalindrome_1(head: ListNode) -> bool:
    stack = []
    slow = fast = head
    while fast is not None and fast.next is not None:
        stack.append(slow.val)
        slow = slow.next
        fast = fast.next.next

    node = slow
    central = True
    while node is not None:
        if len(stack) > 0 and stack[-1] == node.val:
            stack.pop()
            node = node.next
        elif central:
            central = False
            node = node.next
        else:
            return False

    if len(stack) == 0:
        return True

    return False


# Reverse linked lists, T = O(n), S = O(1)
def isPalindrome_2(head: ListNode) -> bool:
    if head is None:
        return True

    slow = fast = head
    pre = pre2 = None
    while fast is not None and fast.next is not None:
        if pre is None:
            pre = slow
        else:
            tmp = pre.next
            pre.next = pre2
            pre2 = pre
            pre = tmp

        slow = slow.next
        fast = fast.next.next

    if pre is not None:
        pre.next = pre2
    central_off = True
    while pre is not None:
        if slow is None:
            return False

        if pre.val != slow.val:
            if central_off:
                slow = slow.next
                central_off = False
            else:
                return False
        else:
            pre = pre.next
            slow = slow.next

    return True


if __name__ == '__main__':
    assert isPalindrome_1(list_to_link([])) is True
    assert isPalindrome_1(list_to_link([1, 2])) is False
    assert isPalindrome_1(list_to_link([1, 2, 2, 1])) is True
    assert isPalindrome_1(list_to_link([1, 3, 1])) is True
    assert isPalindrome_2(list_to_link([])) is True
    assert isPalindrome_2(list_to_link([1, 2])) is False
    assert isPalindrome_2(list_to_link([1, 2, 2, 1])) is True
    assert isPalindrome_2(list_to_link([1, 3, 1])) is True
