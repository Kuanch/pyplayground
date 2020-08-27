seen = []


def exist(board, word):
    global seen
    for r in range(len(board)):
        for c in range(len(board[0])):
            if visit(0, word, board, r, c):
                return True
            else:
                seen = []

    return False


def visit(c, word, board, row, col):
    ans = False
    if (row, col) in seen:
        seen.append((row, col))
        return ans
    seen.append((row, col))

    if word[c] == board[row][col]:
        ans = True
        if c + 1 < len(word):
            ans = False
            if row + 1 < len(board):
                if visit(c + 1, word, board, row + 1, col):
                    ans = True
                else:
                    seen.remove((row + 1, col))

            if row - 1 >= 0:
                if visit(c + 1, word, board, row - 1, col):
                    ans = True
                else:
                    seen.remove((row - 1, col))

            if col + 1 < len(board[0]):
                if visit(c + 1, word, board, row, col + 1):
                    ans = True
                else:
                    seen.remove((row, col + 1))

            if col - 1 >= 0:
                if visit(c + 1, word, board, row, col - 1):
                    ans = True
                else:
                    seen.remove((row, col - 1))

    return ans


if __name__ == '__main__':
    board = [
        ['A', 'B', 'C', 'E'],
        ['S', 'F', 'C', 'S'],
        ['A', 'D', 'E', 'E']
    ]
    assert exist(board, "ABCCED") is True
    assert exist(board, "ABCB") is False
