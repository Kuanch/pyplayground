def merge(intervals):
    merge = []
    intervals.sort()
    for interval in intervals:
        if len(merge) == 0 or merge[-1][1] < interval[0]:
            merge.append(interval)

        else:
            if merge[-1][1] >= interval[1]:
                continue
            else:
                merge[-1][1] = interval[1]

    return merge


if __name__ == '__main__':
    assert merge([[1, 4], [4, 5]]) == [[1, 5]]
    assert merge([[2, 6], [1, 3], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
