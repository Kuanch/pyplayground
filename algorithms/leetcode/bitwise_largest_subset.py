def solution(A):
    return bitwise_and(A)


def bitwise_and(nums):
    bits = 1
    for num in nums:
        bits &= num
    if bits > 0:
        return nums
    else:
        max_subset = bitwise_and(nums[:-1])
        res = nums[-1]
        for n in max_subset:
            res &= n
        if res > 0:
            return max_subset + [nums[-1]]
        else:
            return max_subset


if __name__ == '__main__':
    print(solution([13, 7, 8, 2, 3]))
