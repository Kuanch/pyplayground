class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __eq__(self, other):
        if isinstance(other, ListNode):
            if self.val != other.val:
                return False

            if self.next is not None and other.next is not None:
                self = self.next
                other = other.next

                return self == other

            elif self.next is None and other.next is None:
                return True

            else:
                return False

        return False


def list_to_link(l):
    if len(l) == 0:
        return ListNode(None)
    head = None
    pre = None
    for i in l:
        node = ListNode(i)
        if head is None:
            pre = head = node
            continue
        pre.next = node
        pre = pre.next

    return head


def lists_to_links(lists):
    node_lists = []
    for l in lists:
        node_lists.append(list_to_link(l))

    return node_lists
