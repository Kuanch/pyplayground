def minSubArrayLen(s, nums):
    if sum(nums) < s:
        return 0

    res = 100000  # max_int
    left = 0
    n_sum = 0
    for i in range(len(nums)):
        n_sum += nums[i]
        while left <= i and n_sum >= s:
            res = min(res, i - left + 1)
            n_sum -= nums[left]
            left += 1

    return res


if __name__ == '__main__':
    assert minSubArrayLen(7, [2, 3, 1, 2, 4, 3]) == 2
    assert minSubArrayLen(100, []) == 0
    assert minSubArrayLen(3, [3]) == 1
