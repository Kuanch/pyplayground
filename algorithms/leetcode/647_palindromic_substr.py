def countSubstrings(s: str) -> int:
    ans = 0
    s = list(s)
    for idx, c in enumerate(s):
        ans += helper(s, idx, idx)
        ans += helper(s, idx, idx + 1)

    return ans


def helper(string, start_idx, end_idx):
    if start_idx >= 0 and end_idx < len(string) and string[start_idx] == string[end_idx]:
        return helper(string, start_idx - 1, end_idx + 1) + 1

    return 0


if __name__ == '__main__':
    assert countSubstrings("abc") == 3
    assert countSubstrings("aaa") == 6
