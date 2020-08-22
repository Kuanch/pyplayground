def twoSum(numbers, target):
    i = 0
    j = len(numbers) - 1
    while i < j:
        if numbers[i] + numbers[j] < target:
            i += 1
        elif numbers[i] + numbers[j] > target:
            j -= 1
        else:
            return [i + 1, j + 1]


if __name__ == '__main__':
    assert twoSum([2, 7, 11, 15], 9) == [1, 2]
