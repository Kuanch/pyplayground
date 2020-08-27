from utils.node_utils import list_to_link


def detectCycle(head):
    if not head or not head.next:
        return None

    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            break

    if slow is not fast:
        return None

    fast = head
    while slow is not fast:
        slow = slow.next
        fast = fast.next

    return slow


if __name__ == '__main__':
    head = list_to_link([3, 2, 0, -4])
    head.next.next.next.next = head.next
    assert detectCycle(head) is head.next
