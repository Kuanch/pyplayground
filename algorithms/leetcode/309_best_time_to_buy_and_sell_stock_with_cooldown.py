def maxProfit1(prices):
    buy = [0] * len(prices)
    buy[0] = -prices[0]
    sell = [0] * len(prices)
    rest = [0] * len(prices)

    for idx, p in enumerate(prices):
        if idx > 0:
            buy[idx] = max(rest[idx - 1] - p, buy[idx - 1])
            sell[idx] = max(buy[idx - 1] + p, sell[idx - 1])
            rest[idx] = max(sell[idx - 1], buy[idx - 1], rest[idx - 1])

    print(buy, sell, rest)
    return sell[-1]


def maxProfit2(prices):
    buy = [0] * len(prices)
    buy[0] = -prices[0]
    sell = [0] * len(prices)
    for i, p in enumerate(prices):
        if i > 0:
            if i >= 2:
                buy[i] = max(sell[i - 2] - p, buy[i - 1])
            else:
                buy[i] = buy[i - 1]
            sell[i] = max(buy[i - 1] + p, sell[i - 1])

    print(buy, sell)
    return sell[-1]


def maxProfit3(prices):
    buy = -float('inf')
    pre_buy = pre_sell = sell = 0
    for p in prices:
        pre_buy = buy
        buy = max(pre_sell - p, pre_buy)
        pre_sell = sell
        sell = max(pre_buy + p, pre_sell)

    return sell


if __name__ == '__main__':
    assert maxProfit1([1, 2, 3, 0, 2]) == 3
    assert maxProfit2([1, 2, 3, 0, 2]) == 3
    assert maxProfit3([1, 2, 3, 0, 2]) == 3
