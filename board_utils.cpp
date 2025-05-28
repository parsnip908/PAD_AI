// board_utils.cpp
#include <array>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <algorithm>

constexpr int ROWS = 5;
constexpr int COLS = 6;
using Board = std::array<std::array<int, COLS>, ROWS>;

Board gen_board() {
    static std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, 5);
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

Board swap_adjacent(Board board, const std::array<int, 2>& loc, int direction) {
    int r = loc[0], c = loc[1];
    int dr[4] = {-1, 1, 0, 0};
    int dc[4] = {0, 0, -1, 1};
    int nr = r + dr[direction], nc = c + dc[direction];
    if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS) {
        std::swap(board[r][c], board[nr][nc]);
    }
    return board;
}

int find_match_3_score(const Board& grid) {
    std::vector<std::vector<bool>> visited(ROWS, std::vector<bool>(COLS, false));
    std::vector<std::vector<bool>> matchable(ROWS, std::vector<bool>(COLS, false));

    // Match-3 horizontal and vertical
    for (int i = 0; i < ROWS; ++i)
        for (int j = 0; j < COLS - 2; ++j)
            if (grid[i][j] == grid[i][j+1] && grid[i][j] == grid[i][j+2])
                matchable[i][j] = matchable[i][j+1] = matchable[i][j+2] = true;

    for (int j = 0; j < COLS; ++j)
        for (int i = 0; i < ROWS - 2; ++i)
            if (grid[i][j] == grid[i+1][j] && grid[i][j] == grid[i+2][j])
                matchable[i][j] = matchable[i+1][j] = matchable[i+2][j] = true;

    auto bfs = [&](int i, int j) {
        int count = 1;
        std::queue<std::pair<int,int>> q;
        q.push({i,j});
        visited[i][j] = true;
        int val = grid[i][j];
        while (!q.empty()) {
            auto [x, y] = q.front(); q.pop();
            for (auto [dx, dy] : std::vector<std::pair<int,int>>{{0,1},{1,0},{0,-1},{-1,0}}) {
                int nx = x + dx, ny = y + dy;
                if (nx < 0 || ny < 0 || nx >= ROWS || ny >= COLS) continue;
                if (!visited[nx][ny] && matchable[nx][ny] && grid[nx][ny] == val) {
                    visited[nx][ny] = true;
                    q.push({nx, ny});
                    count++;
                }
            }
        }
        return count;
    };

    int total_score = 0;
    for (int i = 0; i < ROWS; ++i)
        for (int j = 0; j < COLS; ++j)
            if (matchable[i][j] && !visited[i][j]) {
                int group_size = bfs(i, j);
                total_score += (group_size - 2) * 10;
            }

    return total_score;
}
