def coinChange(coins, amount):
    dp = [amount + 1] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for c in coins:
            if c <= i:
                dp[i] = min(dp[i], dp[i - c] + 1)

    if dp[amount] > amount:
        return -1
    else:
        return dp[amount]


if __name__ == '__main__':
    assert coinChange([1, 2, 5], 11) == 3
    assert coinChange([2], 3) == -1
