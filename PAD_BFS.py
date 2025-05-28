import numpy as np
from collections import deque
from board_utils_numpy import gen_board, find_match_3_score, swap_adjacent
import time

# Direction mappings
DIRS = {0: "up", 1: "down", 2: "left", 3: "right"}
# OPPOSITE_DIR = {0: 1, 1: 0, 2: 3, 3: 2}
OPPOSITE_DIR = [1, 0, 3, 2]

def bfs_best_score(initial_board, initial_pos, max_depth=10, verbose=True, prev_dir=None):
    start_time = time.time()

    best_score = find_match_3_score(initial_board)
    best_board = initial_board
    best_action = None
    visited = set()

    queue = deque()
    queue.append((initial_board, initial_pos, 0, prev_dir, None))

    depth_time = {}
    max_reached_depth = -1
    skip_count = 0

    while queue:
        board, pos, depth, prev_dir, first_action = queue.popleft()
        board_key = board.tobytes()
        state_id = (board_key, tuple(pos))

        if state_id in visited:
            skip_count += 1
            continue
        visited.add(state_id)

        # Track time when first reaching a new depth
        if depth > max_reached_depth:
            depth_time[depth] = time.time() - start_time
            if verbose:
                print(f"Reached depth {depth:>2} at {depth_time[depth]:.4f} seconds and skipped {skip_count:>6} states")
            max_reached_depth = depth

        if depth == max_depth:
            score = find_match_3_score(board)
            if score > best_score:
                best_score = score
                best_board = board
                best_action = first_action
            continue

        for direction in range(4):
            if prev_dir is not None and direction == OPPOSITE_DIR[prev_dir]:
                continue  # Avoid undoing last move

            new_board = swap_adjacent(board.copy(), pos, direction)
            new_pos = pos.copy()
            if direction == 0 and pos[0] > 0: new_pos[0] -= 1
            elif direction == 1 and pos[0] < 4: new_pos[0] += 1
            elif direction == 2 and pos[1] > 0: new_pos[1] -= 1
            elif direction == 3 and pos[1] < 5: new_pos[1] += 1
            else: continue

            next_first_action = direction if depth == 0 else first_action
            queue.append((new_board, new_pos, depth + 1, direction, next_first_action))

    total_time = time.time() - start_time
    if verbose:
        print(f"Total time taken: {total_time:.4f} seconds")
    return best_score, best_board, DIRS[best_action] if best_action is not None else None


test_board = [[5, 2, 4, 5, 5, 1],
              [3, 3, 5, 0, 5, 3],
              [5, 2, 0, 3, 3, 4],
              [1, 0, 1, 1, 5, 2],
              [5, 3, 4, 1, 4, 2]]
test_pos = [3, 2]

def play_game_with_bfs(max_moves=10, search_depth=10, verbose=True):
    board, pos = gen_board()
    # board, pos = np.array(test_board), np.array(test_pos)


    move_count = 0

    initial_state = (board, pos)
    direction = None

    if verbose:
        print("Initial Board:")
        print(board)

    while move_count < max_moves:
        if search_depth > (max_moves-move_count):
            search_depth = max_moves-move_count
            if verbose:
                print(f"Search Depth is now {search_depth}")

        score, _, action = bfs_best_score(board, pos, prev_dir=direction, max_depth=search_depth, verbose=False)
        if action is None:
            if verbose:
                print("No beneficial action found. Ending early.")
            break

        # Apply the chosen action
        direction = {"up": 0, "down": 1, "left": 2, "right": 3}[action]
        board = swap_adjacent(board, pos, direction)

        # Update position
        if direction == 0 and pos[0] > 0: pos[0] -= 1
        elif direction == 1 and pos[0] < 4: pos[0] += 1
        elif direction == 2 and pos[1] > 0: pos[1] -= 1
        elif direction == 3 and pos[1] < 5: pos[1] += 1

        move_count += 1

        if verbose:
            print(f"\nMove {move_count}: {action}")
            print(board)
            print(f"Current Match-3 Score: {find_match_3_score(board)}")
            print(f"Projected Match-3 Score: {score}")

    final_score = find_match_3_score(board)
    if verbose:
        print("\n\nInitial Board:")
        print("Position:", initial_state[1])
        print(initial_state[0])
        print("\n\nFinal Board:")
        print(board)
        print(f"\nFinal Match-3 Score: {final_score}")

    return board, final_score
