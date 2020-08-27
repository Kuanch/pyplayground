def decodeString(s: str) -> str:
    char_cnt = ''
    int_cnt = ''
    int_stack = []
    char_stack = []
    s = list(s)
    for i in s:
        if i.isnumeric():
            int_cnt += i
        elif i == '[':
            int_stack.append(int(int_cnt))
            char_stack.append(char_cnt)
            int_cnt = ''
            char_cnt = ''
        elif i == ']':
            char_cnt = char_stack.pop() + char_cnt * int_stack.pop()
        else:
            char_cnt += i

    return char_cnt


if __name__ == '__main__':
    assert decodeString("3[a]2[bc]") == "aaabcbc"
    assert decodeString("3[a2[c]]") == "accaccacc"
    assert decodeString("2[abc]3[cd]ef") == "abcabccdcdcdef"
    assert decodeString("abc3[cd]xyz") == "abccdcdcdxyz"
