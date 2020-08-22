def validPalindrome(s: str) -> bool:
    return check_palindrome(s, False)


def check_palindrome(s, removed):
    begin = 0
    end = len(s) - 1
    while begin <= end:
        if s[begin] == s[end]:
            begin += 1
            end -= 1
        elif not removed:
            if check_palindrome(s[begin + 1:end + 1], True):
                return True
            elif check_palindrome(s[begin:end], True):
                return True
            else:
                return False
        else:
            return False

    return True


if __name__ == '__main__':
    assert validPalindrome('aba') is True
    assert validPalindrome('abbb') is True
    assert validPalindrome('abbbb') is True
    assert validPalindrome('abbbc') is False
