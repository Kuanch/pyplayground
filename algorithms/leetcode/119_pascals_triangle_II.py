def getRow(rowIndex):
    pre = init = [1, 1]
    if rowIndex == 0:
        return [1]
    if rowIndex == 1:
        return init

    for r in range(2, rowIndex + 1):
        nxt = [1]
        for i in range(1, len(pre) + 1):
            if i < r:
                nxt.append(pre[i - 1] + pre[i])
            else:
                nxt.append(1)

        pre = nxt

    return nxt


if __name__ == '__main__':
    assert getRow(0) == [1]
    assert getRow(1) == [1, 1]
    assert getRow(4) == [1, 4, 6, 4, 1]
