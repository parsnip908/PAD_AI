from collections import deque
from collections import defaultdict
import torch

score_table = {
    3:  10,
    4:  20,
    5:  40,
    6:  70,
    7:  100,
    8:  130,
    9:  170,
    10: 210,
    11: 250,
    12: 300
}

def find_match_3_score(grid):
    rows, cols = len(grid), len(grid[0])

    def get_score(length):
        max_len = max(score_table)
        return score_table.get(length, score_table[max_len])

    visited = [[False] * cols for _ in range(rows)]
    matchable = [[False] * cols for _ in range(rows)]

    # Step 1: Identify all individual match-3+ locations
    for i in range(rows):
        for j in range(cols - 2):
            if grid[i][j] == grid[i][j + 1] == grid[i][j + 2]:
                matchable[i][j] = matchable[i][j + 1] = matchable[i][j + 2] = True
    for j in range(cols):
        for i in range(rows - 2):
            if grid[i][j] == grid[i + 1][j] == grid[i + 2][j]:
                matchable[i][j] = matchable[i + 1][j] = matchable[i + 2][j] = True

    # Step 2: Flood-fill to group connected matchable regions
    def bfs(start_i, start_j):
        q = deque()
        q.append((start_i, start_j))
        visited[start_i][start_j] = True
        val = grid[start_i][start_j]
        count = 1

        while q:
            i, j = q.popleft()

            # Check right
            ni, nj = i, j + 1
            if nj < cols and not visited[ni][nj] and matchable[ni][nj] and grid[ni][nj] == val:
                visited[ni][nj] = True
                q.append((ni, nj))
                count += 1

            # Check down
            ni, nj = i + 1, j
            if ni < rows and not visited[ni][nj] and matchable[ni][nj] and grid[ni][nj] == val:
                visited[ni][nj] = True
                q.append((ni, nj))
                count += 1

        return count

    total_score = 0
    for i in range(rows):
        for j in range(cols):
            if matchable[i][j] and not visited[i][j]:
                group_size = bfs(i, j)
                total_score += (group_size-2)*10 #get_score(group_size)

    return total_score


def proximity_score(grid):

    rows, cols = len(grid), len(grid[0])
    value_positions = defaultdict(list)

    # Collect positions for each value
    for i in range(rows):
        for j in range(cols):
            value_positions[grid[i][j].item()].append((i, j))

    def manhattan(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    # print(value_positions)

    score = 0.0
    for positions in value_positions.values():
        n = len(positions)
        if n < 2:
            continue
        for i in range(n):
            for j in range(i + 1, n):
                dist = manhattan(positions[i], positions[j])
                score += 1.0 / (dist + 1)  # closer tiles contribute more

    return score


def swap_adjacent(board, location, direction):
    """
    Swaps the value at the given location with an adjacent cell in the specified direction.

    Parameters:
    - board: torch.Tensor of shape (5, 6)
    - location: torch.Tensor of shape (>=2,) where location[0]=row, location[1]=col
    - direction: int, one of {0=up, 1=down, 2=left, 3=right}

    Returns:
    - torch.Tensor: a new tensor with the swap applied, or the same board if invalid.
    """
    row, col = int(location[0].item()), int(location[1].item())
    swap_row, swap_col = row, col

    if direction == 0:       # up
        swap_row -= 1
    elif direction == 1:     # down
        swap_row += 1
    elif direction == 2:     # left
        swap_col -= 1
    elif direction == 3:     # right
        swap_col += 1
    else:
        return board  # Invalid direction

    if 0 <= swap_row < 5 and 0 <= swap_col < 6:
        with torch.no_grad():
            board = board.clone()
            temp = board[row, col].item()
            board[row, col] = board[swap_row, swap_col]
            board[swap_row, swap_col] = temp

    return board
