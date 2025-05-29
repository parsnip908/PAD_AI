// board_utils.cpp

#include <array>
#include <vector>
#include <queue>
#include <random>
#include <algorithm>

constexpr int ROWS = 5;
constexpr int COLS = 6;
constexpr int COLORS = 6;
using Board = std::array<std::array<int, COLS>, ROWS>;

constexpr std::array<int, 4> dr = {-1, 1, 0, 0};
constexpr std::array<int, 4> dc = {0, 0, -1, 1};

Board gen_board() {
    static std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, COLORS - 1);
    Board board;
    for (auto& row : board)
        for (auto& cell : row)
            cell = dist(rng);
    return board;
}

std::array<int, 2> gen_position() {
    static std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist_row(0, ROWS - 1);
    std::uniform_int_distribution<int> dist_col(0, COLS - 1);
    return {dist_row(rng), dist_col(rng)};
}

// direction: 0 = up, 1 = down, 2 = left, 3 = right
Board swap_adjacent(Board board, const std::array<int, 2>& loc, int direction) {
    int r = loc[0], c = loc[1];
    int nr = r + dr[direction], nc = c + dc[direction];
    if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS) {
        std::swap(board[r][c], board[nr][nc]);
    }
    return board;
}

// Returns score for all match-3 (and larger) groups.
// Matches are found as in Puzzles & Dragons: adjacent matches are merged.
int find_match_3_score(const Board& grid) {
    std::vector<std::vector<bool>> visited(ROWS, std::vector<bool>(COLS, false));
    std::vector<std::vector<bool>> matchable(ROWS, std::vector<bool>(COLS, false));

    // Mark all horizontal and vertical match-3 locations as matchable.
    for (int i = 0; i < ROWS; ++i)
        for (int j = 0; j < COLS - 2; ++j)
            if (grid[i][j] == grid[i][j+1] && grid[i][j] == grid[i][j+2])
                matchable[i][j] = matchable[i][j+1] = matchable[i][j+2] = true;

    for (int j = 0; j < COLS; ++j)
        for (int i = 0; i < ROWS - 2; ++i)
            if (grid[i][j] == grid[i+1][j] && grid[i][j] == grid[i+2][j])
                matchable[i][j] = matchable[i+1][j] = matchable[i+2][j] = true;

    int total_score = 0;
    std::queue<std::pair<int,int>> q;
    for (int i = 0; i < ROWS; ++i)
        for (int j = 0; j < COLS; ++j)
            if (matchable[i][j] && !visited[i][j])
            {
                // Group connected matchable cells using BFS.
                int group_size = 1;
                q.push({i, j});
                visited[i][j] = true;
                int val = grid[i][j];
                while (!q.empty())
                {
                    auto [x, y] = q.front(); q.pop();
                    for (int d = 0; d < 4; ++d)
                    {
                        int nx = x + dr[d], ny = y + dc[d];
                        if (nx < 0 || ny < 0 || nx >= ROWS || ny >= COLS)
                            continue;
                        if (!visited[nx][ny] && matchable[nx][ny] && grid[nx][ny] == val) {
                            visited[nx][ny] = true;
                            q.push({nx, ny});
                            ++group_size;
                        }
                    }
                }
                total_score += (group_size - 2) * 10;
            }
    return total_score;
}
