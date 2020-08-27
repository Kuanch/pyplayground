def findNumberOfLIS(nums):
    res = 0                   # number of leng of longest seq.
    max_l = 0                 # leng of longest seq.
    cnt = [1] * len(nums)     # current number of leng of longest seq.
    length = [1] * len(nums)  # leng of longest seq.
    for i in range(len(nums)):
        for j in range(len(nums)):
            if j >= i:
                break

            # [2, 2, 2, 2, 2]
            if nums[i] <= nums[j]:
                continue

            if length[i] == length[j] + 1:
                cnt[i] += cnt[j]

            # Find another longer increasing seq.
            elif length[i] < length[j] + 1:
                length[i] = length[j] + 1
                cnt[i] = cnt[j]

        if max_l == length[i]:
            res += cnt[i]
        elif max_l < length[i]:
            res = cnt[i]
            max_l = length[i]

    return res


if __name__ == '__main__':
    assert findNumberOfLIS([1, 3, 5, 4, 7]) == 2
