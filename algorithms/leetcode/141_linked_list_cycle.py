from utils.node_utils import list_to_link


def hasCycle(head):
    if head is None:
        return False
    slow = fast = head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next

        '''
        is operator is used to compare 2 object
        the same type and value
        '''
        if slow is fast:
            return True

    return False


if __name__ == '__main__':
    head1 = list_to_link([3, 2, 0, -1])
    head1.next.next.next.next = head1.next  # cycle postion at node1
    assert hasCycle(head1) is True
