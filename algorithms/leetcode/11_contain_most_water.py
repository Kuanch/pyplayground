def maxArea(height) -> int:
    max_area = 0
    start = 0
    end = len(height) - 1
    while start < end:
        max_area = max(min(height[start], height[end]) * abs(start - end), max_area)
        if height[start] > height[end]:
            end -= 1
        else:
            start += 1

    return max_area


if __name__ == '__main__':
    assert maxArea([1, 8, 6, 2, 5, 4, 8, 3, 7]) == 49
