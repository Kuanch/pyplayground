def maxProfit1(prices):
    g = []
    l = []
    for i in range(len(prices)):
        g.append([0] * 4)
        l.append([0] * 4)
    for i, p in enumerate(prices):
        if i == 0:
            diff = 0
        else:
            diff = p - prices[i - 1]
            for j in range(1, 4):
                l[i][j] = max(g[i - 1][j - 1], l[i - 1][j]) + diff
                g[i][j] = max(l[i][j], g[i - 1][j])

    return g[-1][2]


def maxProfit2(prices):
    l = [0] * 3
    g = [0] * 3
    for i, p in enumerate(prices):
        if i == 0:
            diff = 0
        else:
            diff = p - prices[i - 1]
            for j in range(2, 0, -1):
                l[j] = max(g[j - 1], l[j]) + diff
                g[j] = max(l[j], g[j])

    return g[2]


if __name__ == '__main__':
    assert maxProfit1([1, 2, 3, 4, 5]) == 4
    assert maxProfit1([3, 3, 5, 0, 0, 3, 1, 4]) == 6
    assert maxProfit1([7, 6, 4, 3, 1]) == 0
    assert maxProfit2([1, 2, 3, 4, 5]) == 4
    assert maxProfit2([3, 3, 5, 0, 0, 3, 1, 4]) == 6
    assert maxProfit2([7, 6, 4, 3, 1]) == 0
