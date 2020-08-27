def numIslands(grid):
    def count_island(row, col, grid, visited):
        if (row, col) not in visited:
            visited.append((row, col))
            if grid[row][col] == '1':
                if row < len(grid) - 1:
                    count_island(row + 1, col, grid, visited)
                if col < len(grid[0]) - 1:
                    count_island(row, col + 1, grid, visited)
                if row - 1 >= 0:
                    count_island(row - 1, col, grid, visited)
                if col - 1 >= 0:
                    count_island(row, col - 1, grid, visited)

                return 1

        return 0

    visited = list()
    area_island = 0
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] == '0':
                visited.append((row, col))
                continue
            area_island += count_island(row, col, grid, visited)

    return area_island


def main():
    case1 = [["1", "1", "1", "1", "0"],
             ["1", "1", "0", "0", "0"],
             ["1", "0", "1", "1", "0"],
             ["0", "0", "0", "0", "1"]]
    assert 3 == numIslands(case1)


if __name__ == '__main__':
    main()
