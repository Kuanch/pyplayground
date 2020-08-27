def threeSum(nums):
    ans = []
    nums = sorted(nums)
    for num_idx, num in enumerate(nums):
        if num > 0 or num_idx >= len(nums) - 2:
            break
        if num_idx > 0 and nums[num_idx - 1] == num:
            continue

        i = num_idx + 1
        j = len(nums) - 1

        while i < j:
            sum_3 = nums[i] + nums[j] + num
            if sum_3 > 0:
                j -= 1
            elif sum_3 < 0:
                i += 1
            elif sum_3 == 0:
                ans.append([num, nums[i], nums[j]])
                while i < j and nums[i] == nums[i + 1]:
                    i += 1
                while i < j and nums[j] == nums[j - 1]:
                    j -= 1
                i += 1
                j -= 1

    return ans


if __name__ == '__main__':
    assert threeSum([-1, 0, 1, 2, -1, -4]) == [[-1, -1, 2], [-1, 0, 1]]
    assert threeSum([0, 0, 0, 0]) == [[0, 0, 0]]
    assert threeSum([-2, 0, 0, 2, 2]) == [[-2, 0, 2]]
