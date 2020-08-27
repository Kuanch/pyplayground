def maxScore(self, s: str) -> int:
    zeros = 1 if s[0] == '0' else 0
    ones = s.count('1', 1)
    ans = ones + zeros
    for idx, i in enumerate(s[1:]):
        if i == '0' and idx + 1 < len(s) - 1:
            zeros += 1
        else:
            ones -= 1

        ans = max(ans, ones + zeros)

    return ans


if __name__ == '__main__':
    assert maxScore('011101') == 5
    assert maxScore('00111') == 5
    assert maxScore('00') == 1
    assert maxScore('11') == 1
    assert maxScore('1111') == 3
    assert maxScore('01') == 2
