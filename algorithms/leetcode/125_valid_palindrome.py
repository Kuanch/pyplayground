def isPalindrome(s: str) -> bool:
    beg = 0
    end = len(s) - 1
    while beg <= end:
        while not s[beg].isalnum() and beg < end:
            beg += 1
        while not s[end].isalnum() and beg < end:
            end -= 1
        if s[beg].lower() == s[end].lower():
            beg += 1
            end -= 1
        else:
            return False

    return True


if __name__ == '__main__':
    assert isPalindrome('A man, a plan, a canal: Panama') is True
    assert isPalindrome('race a car') is False
