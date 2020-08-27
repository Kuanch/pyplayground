def findLength(A, B):
    dp = [[0] * (len(B) + 1) for _ in range(len(A) + 1)]

    for i in range(len(A)):
        for j in range(len(B)):
            if A[i] == B[j]:
                dp[i][j] = dp[i - 1][j - 1] + 1

    return max([max(row) for row in dp])


if __name__ == '__main__':
    assert findLength([1, 2, 3, 2, 1], [3, 2, 1, 4, 7]) == 3
