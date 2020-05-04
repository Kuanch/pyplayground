def quick_sort(l):
    def divide(l):
        if len(l) == 0:
            return []
        elif len(l) == 1:
            return l
        i = -1
        j = 0
        pivot = l[-1]
        while j < len(l):
            if l[j] < pivot:
                temp = l[i + 1]
                l[i + 1] = l[j]
                l[j] = temp
                i += 1
            j += 1

        temp = l[i + 1]
        l[i + 1] = l[-1]
        l[-1] = temp

        return divide(l[:i + 1]) + l[i + 1:i + 2] + divide(l[i + 2:])
    return divide(l)


def main():
    input_l = [14, 23, 90, 97, 47, 35, 10, 78, 37, 81, 33]
    qsort_l = quick_sort(input_l)
    input_l.sort()

    assert input_l == qsort_l


if __name__ == '__main__':
    main()
