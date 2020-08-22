def twoSum(nums, target):
    mapping = {}
    for idx, num in enumerate(nums):
        if len(mapping) == 0:
            mapping[num] = idx
        else:
            if target - num in mapping:
                return [mapping[target - num], idx]
            else:
                mapping[num] = idx

    return []


if __name__ == '__main__':
    assert twoSum([2, 7, 11, 15], 9) == [0, 1]
