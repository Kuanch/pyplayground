def merge_sort(l):
    def divide(l):
        if len(l) == 1:
            return l

        elif len(l) == 2:
            if l[0] > l[1]:
                temp = l[1]
                l[1] = l[0]
                l[0] = temp

            return l

        elif len(l) > 2:
            half_leng = int(len(l) / 2)
            return msort(divide(l[:half_leng + 1]), divide(l[half_leng + 1:]))

    def msort(l1, l2):
        l = []
        leng1 = len(l1)
        leng2 = len(l2)
        idx1 = 0
        idx2 = 0
        while idx1 <= leng1 - 1 and idx2 <= leng2 - 1:
            if l1[idx1] < l2[idx2]:
                l.append(l1[idx1])
                idx1 += 1
            elif l1[idx1] > l2[idx2]:
                l.append(l2[idx2])
                idx2 += 1

            else:
                l.append(l1[idx1])
                idx1 += 1

        if idx1 > len(l1) - 1:
            l.extend(l2[idx2:])

        elif idx2 > len(l2) - 1:
            l.extend(l1[idx1:])

        return l

    return divide(l)


def main():
    input_l = [14, 23, 90, 97, 47, 35, 10, 78, 37, 81]
    msort_l = merge_sort(input_l)
    input_l.sort()

    assert input_l == msort_l


if __name__ == '__main__':
    main()
