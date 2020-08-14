from utils.node_utils import lists_to_links, list_to_link


def mergeKLists(lists):
    if len(lists) > 1:
        merge_list = compare_merge(lists[0], lists[1])
        for i in range(2, len(lists)):
            merge_list = compare_merge(merge_list, lists[i])

        return merge_list

    elif len(lists) == 1:
        return lists[0]


def compare_merge(l1, l2):
    if l1 is None or l2 is None:
        if l1 is None and l2 is None:
            return None
        if l1 is None:
            return l2
        elif l2 is None:
            return l1

    if l1.val < l2.val:
        l1.next = compare_merge(l1.next, l2)
        return l1
    else:
        l2.next = compare_merge(l2.next, l1)
        return l2


if __name__ == '__main__':
    node_lists = lists_to_links([[1, 4, 5], [1, 3, 4], [2, 6]])
    assert list_to_link([1, 1, 2, 3, 4, 4, 5, 6]) == mergeKLists(node_lists)
