def leastInterval(tasks, n) -> int:
    type_task = set(tasks)
    num_most_task = 0
    deplicate_num_task = 0
    for t in type_task:
        task_count = tasks.count(t)
        if num_most_task < task_count:
            num_most_task = task_count
            deplicate_num_task = 0
        elif task_count == num_most_task:
            deplicate_num_task += 1

    return max(len(tasks), (num_most_task - 1) * (n + 1) + deplicate_num_task + 1)


if __name__ == '__main__':
    assert leastInterval(["A", "A", "A", "B", "B", "B"], 2) == 8
    assert leastInterval(["A", "A", "A", "A", "A", "A",
                          "B", "C", "D", "E", "F", "G"], 2) == 16
