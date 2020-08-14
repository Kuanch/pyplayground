from utils.node_utils import list_to_link


def getIntersectionNode(headA, headB):
    n1 = n2 = 0
    node1 = headA
    node2 = headB
    while node1 or node2:
        if node1:
            node1 = node1.next
            n1 += 1
        if node2:
            node2 = node2.next
            n2 += 1
            
    diff = abs(n1 - n2)
    if n1 > n2:
        for i in range(diff):
            headA = headA.next
    else:
        for i in range(diff):
            headB = headB.next
            
    while headA is not headB:
        headA = headA.next
        headB = headB.next
    
    return headA


if __name__ == '__main__':
    l1 = list_to_link([4, 1])
    l2 = list_to_link([5, 6, 1])
    intersection = list_to_link([8, 4, 5])
    l1.next.next = intersection
    l2.next.next.next = intersection
    assert getIntersectionNode(l1, l2) is intersection
