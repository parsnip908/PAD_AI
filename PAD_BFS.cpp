// pad_bfs.cpp

#include <iostream>
#include <queue>
#include <unordered_set>
#include <chrono>
#include <tuple>
#include <string>
#include <sstream>
#include <optional>
#include "board_utils.cpp" // Prefer .h/.cpp split, but .cpp is fine for quick builds.

struct StateHash {
    std::size_t operator()(const std::pair<Board, std::array<int, 2>>& s) const {
        std::size_t h = 0;
        for (const auto& row : s.first)
            for (int val : row)
                h ^= std::hash<int>{}(val) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(s.second[0]) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(s.second[1]) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

const std::array<std::string, 4> DIRS = {"up", "down", "left", "right"};
const std::array<int, 4> OPPOSITE_DIR = {1, 0, 3, 2};
// constexpr std::array<int, 4> dr = {-1, 1, 0, 0};
// constexpr std::array<int, 4> dc = {0, 0, -1, 1};

void bfs_best_score(const Board& initial_board, const std::array<int, 2>& initial_pos, int max_depth) {
    using Clock = std::chrono::high_resolution_clock;
    auto start_time = Clock::now();

    int best_score = 0;
    Board best_board = initial_board;
    std::optional<int> best_action;

    std::unordered_set<std::pair<Board, std::array<int, 2>>, StateHash> visited;
    std::queue<std::tuple<Board, std::array<int, 2>, int, std::optional<int>, std::optional<int>>> q;
    q.push({initial_board, initial_pos, 0, std::nullopt, std::nullopt});

    int max_depth_reached = -1;
    int skip_count = 0;

    while (!q.empty()) {
        auto [board, pos, depth, prev_dir, first_action] = q.front(); q.pop();

        if (!visited.insert({board, pos}).second) {
            ++skip_count;
            continue;
        }

        if (depth > max_depth_reached) {
            auto now = Clock::now();
            std::chrono::duration<double> elapsed = now - start_time;
            std::cout << "Reached depth " << depth << " at " << elapsed.count() << " seconds, skipped " << skip_count << " states\n";
            max_depth_reached = depth;
        }

        if (depth == max_depth) {
            int score = find_match_3_score(board);
            if (score > best_score) {
                best_score = score;
                best_board = board;
                best_action = first_action;
            }
            continue;
        }

        for (int dir = 0; dir < 4; ++dir) {
            if (prev_dir && dir == OPPOSITE_DIR[*prev_dir]) continue;

            int nr = pos[0] + dr[dir];
            int nc = pos[1] + dc[dir];
            if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS) continue;

            Board new_board = swap_adjacent(board, pos, dir);
            std::array<int, 2> new_pos = {nr, nc};
            q.push({new_board, new_pos, depth + 1, dir, depth == 0 ? std::optional<int>{dir} : first_action});
        }
    }

    auto total_time = Clock::now() - start_time;
    std::cout << "Total time: " << std::chrono::duration<double>(total_time).count() << " seconds\n";
    std::cout << "Best score: " << best_score << ", First action: "
              << (best_action ? DIRS[*best_action] : "None") << "\n";

    std::cout << "Best board:\n";
    for (const auto& row : best_board) {
        for (int v : row) std::cout << v << " ";
        std::cout << "\n";
    }
}

int main() {
    int depth;
    std::cout << "Enter search depth: ";
    std::cin >> depth;

    Board board = gen_board();
    auto pos = gen_position();

    std::cout << "Initial board:\n";
    for (const auto& row : board) {
        for (int v : row) std::cout << v << " ";
        std::cout << "\n";
    }
    std::cout << "Start position: (" << pos[0] << ", " << pos[1] << ")\n";

    bfs_best_score(board, pos, depth);
    return 0;
}
