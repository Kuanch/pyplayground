def longestCommonSubsequence(text1: str, text2: str) -> int:
    if len(text1) == 0 or len(text2) == 0:
        return 0
    text1 = list(text1)
    text2 = list(text2)

    return len(LCS(text1, text2))


def LCS(t1, t2):
    if len(t1) >= 1 and len(t2) >= 1:
        if t1[-1] == t2[-1]:
            comm_str = LCS(t1[:-1], t2[:-1]) + [t1[-1]]
        else:
            comm_str1 = LCS(t1, t2[:-1])
            comm_str2 = LCS(t1[:-1], t2)
            if len(comm_str1) > len(comm_str2):
                comm_str = comm_str1
            else:
                comm_str = comm_str2
        return comm_str

    else:
        return []


if __name__ == '__main__':
    assert longestCommonSubsequence("abcde", "ace") == 3
    assert longestCommonSubsequence("abc", "abc") == 3
    assert longestCommonSubsequence("abc", "def") == 0
