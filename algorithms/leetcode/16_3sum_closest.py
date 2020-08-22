def threeSumClosest(nums, target: int) -> int:
    closest = float('inf')
    nums = sorted(nums)
    for num_idx, num in enumerate(nums):
        if num_idx > len(nums) - 2:
            break
        i = num_idx + 1
        j = len(nums) - 1
        while i < j:
            diff = abs(nums[i] + nums[j] + num - target)
            smallest_diff = abs(closest - target)
            if diff < smallest_diff:
                closest = nums[i] + nums[j] + num

            if nums[i] + nums[j] + num > target:
                j -= 1
            elif nums[i] + nums[j] + num < target:
                i += 1
            else:
                return closest

    return closest


if __name__ == '__main__':
    assert threeSumClosest([-1, 2, 1, -4], 1) == 2
