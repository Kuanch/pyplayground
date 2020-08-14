def wordBreak(s, wordDict):
    if s in wordDict:
        return True

    dp = []
    for i in range(len(s) + 1):
        dp.append(False)
    dp[0] = True
    for i in range(1, len(s) + 1):
        for j in range(i):
            if (s[j:i] in wordDict) and dp[j]:
                dp[i] = True
                break

    return dp[-1]


if __name__ == '__main__':
    assert wordBreak("leetcode", ["leet", "code"]) is True
    assert wordBreak("applepenapple", ["apple", "pen"]) is True
    assert wordBreak("ab", ["a", "b"]) is True
