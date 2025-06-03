import numpy as np
from collections import deque
from board_utils_numpy import gen_board, find_match_3_score, swap_adjacent, find_max_score
import time

# Direction mappings
DIRS = {0: "up", 1: "down", 2: "left", 3: "right"}
# OPPOSITE_DIR = {0: 1, 1: 0, 2: 3, 3: 2}
OPPOSITE_DIR = [1, 0, 3, 2]

gamma = 0.99

def bfs_best_score(initial_board, initial_pos, max_depth=10, verbose=True, prev_dir=None, reject_dirs=None):
    start_time = time.time()

    # initialize scores
    initial_score = find_match_3_score(initial_board)
    best_score = initial_score
    best_board = initial_board
    best_action = None
    max_score = find_max_score(initial_board)
    Q_values = np.array([float(-initial_score)]*4)
    
    # initialize visited set
    visited = set()
    board_key = initial_board.astype(np.uint8).tobytes()
    pos_key = initial_pos.astype(np.uint8).tobytes()
    state_id = board_key + pos_key
    visited.add(state_id)

    # Initialize Queue
    queue = deque()

    for direction in range(4):
        # eliminate directions forbidden by function inputs
        if prev_dir is not None and direction == OPPOSITE_DIR[prev_dir]:
            continue  # Avoid undoing last move
        if reject_dirs is not None and direction in reject_dirs:
            continue  # Avoid cycle

        # generate new position, check if direction legal
        new_pos = initial_pos.copy()
        if   direction == 0 and initial_pos[0] > 0: new_pos[0] -= 1
        elif direction == 1 and initial_pos[0] < 4: new_pos[0] += 1
        elif direction == 2 and initial_pos[1] > 0: new_pos[1] -= 1
        elif direction == 3 and initial_pos[1] < 5: new_pos[1] += 1
        else: 
            Q_values[direction] = -20
            continue # illegal move
        
        # generate new board
        new_board = swap_adjacent(initial_board, initial_pos, direction)

        # add to visisted set
        board_key = new_board.astype(np.uint8).tobytes()
        pos_key = new_pos.astype(np.uint8).tobytes()
        state_id = board_key + pos_key
        visited.add(state_id)

        # add to queue
        queue.append((new_board, new_pos, 1, direction, direction))

    # init profiling
    depth_time = {}
    max_reached_depth = 0
    skip_count = 0

    # main loop
    while queue:
        board, pos, depth, prev_dir, first_action = queue.popleft()

        # Track time when first reaching a new depth
        if depth > max_reached_depth:
            depth_time[depth] = time.time() - start_time
            if verbose:
                print(f"Reached depth {depth:>2} at {depth_time[depth]:.4f} seconds and skipped {skip_count:>6} states")
                # print(state_id)
            max_reached_depth = depth

        # score board and update metrics
        score = find_match_3_score(board)
        Q = (score - initial_score) * (gamma**(depth-1))
        if Q > Q_values[first_action]:
            Q_values[first_action] = Q
        if score > best_score:
            best_score = score
            best_board = board
            best_action = first_action
            if score >= max_score:
                print("found best state")
                break
        
        # stop expansion at max depth
        if depth == max_depth:
            continue

        # expand search
        for direction in range(4):
            if direction == OPPOSITE_DIR[prev_dir]:
                continue  # Avoid undoing last move

            # generate new state
            new_pos = pos.copy()
            if direction == 0 and pos[0] > 0: new_pos[0] -= 1
            elif direction == 1 and pos[0] < 4: new_pos[0] += 1
            elif direction == 2 and pos[1] > 0: new_pos[1] -= 1
            elif direction == 3 and pos[1] < 5: new_pos[1] += 1
            else: continue
            new_board = swap_adjacent(board, pos, direction)

            # check if new state was visited
            board_key = new_board.astype(np.uint8).tobytes()
            pos_key = new_pos.astype(np.uint8).tobytes()
            state_id = board_key + pos_key

            if state_id in visited:
                skip_count += 1
                continue

            # add to set and queue
            visited.add(state_id)
            queue.append((new_board, new_pos, depth + 1, direction, first_action))

    total_time = time.time() - start_time
    if verbose:
        print(f"Total time taken: {total_time:.4f} seconds")
        print(Q_values)
    return best_score, best_board, best_action, Q_values #if best_action is not None else None


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
    move_path = []
    # prev_dir = None
    max_score = 0
    roaming_mod = 0
    curr_score = find_match_3_score(board)
    score_potential = find_max_score(board)

    if verbose:
        print("Initial Board:")
        print(board)
        print(f"max score: {score_potential}")
        time.sleep(1)


    visited = set()
    board_key = board.astype(np.uint8).tobytes()
    pos_key = pos.astype(np.uint8).tobytes()
    state_id = board_key + pos_key
    visited.add(state_id)

    reject_dirs = []

    while move_count < max_moves:
        if search_depth >= (max_moves-move_count):
            if not roaming_mod:
                if verbose:
                    print("------------------")
                    print("final moves.")
                    print("turning off roaming.\nreseting visited set.")
                    print("------------------\n")
                    time.sleep(1)
                roaming_mod = 0.1
                visited = set()
                reject_dirs = []
            search_depth = max_moves-move_count
            if verbose:
                print(f"Search Depth is now {search_depth}")

        curr_score = find_match_3_score(board)
        if curr_score > max_score:
            max_score = curr_score

        proj_score, _, direction, Q_values = bfs_best_score(board, pos, max_depth=search_depth, verbose=False) #, prev_dir=prev_dir, reject_dirs=reject_dirs
        Q_filtered = Q_values.copy()
        test_dir = True
        
        while test_dir:
            Q_filtered[reject_dirs] = -20
            # direction = np.argmax(Q_filtered)
            direction = np.random.choice(np.flatnonzero(Q_filtered == Q_filtered.max()))
            if Q_filtered[direction] - roaming_mod < 0:
                if (curr_score < max_score or curr_score < proj_score) and not roaming_mod:
                    if verbose:
                        print("------------------")
                        print("roaming failure.")
                        print("turning off roaming.\nreseting visited set.")
                        print("------------------\n")
                        time.sleep(1)
                    roaming_mod = 0.1
                    Q_filtered = Q_values.copy()
                    reject_dirs = []
                    visited = set()
                    board_key = board.astype(np.uint8).tobytes()
                    pos_key = pos.astype(np.uint8).tobytes()
                    state_id = board_key + pos_key
                    visited.add(state_id)
                    continue
                direction = None
                break

            # Apply the chosen action
            # direction = {"up": 0, "down": 1, "left": 2, "right": 3}[action]
            new_board = swap_adjacent(board, pos, direction)

            # Update position
            new_pos = pos.copy()
            if   direction == 0 and pos[0] > 0: new_pos[0] -= 1
            elif direction == 1 and pos[0] < 4: new_pos[0] += 1
            elif direction == 2 and pos[1] > 0: new_pos[1] -= 1
            elif direction == 3 and pos[1] < 5: new_pos[1] += 1

            board_key = new_board.astype(np.uint8).tobytes()
            pos_key = new_pos.astype(np.uint8).tobytes()
            state_id = board_key + pos_key

            if state_id in visited:
                test_dir = True
                reject_dirs.append(direction)
                # if verbose:
                #     print(f"direction {DIRS[direction]} rejected")
                continue
            visited.add(state_id)
            test_dir = False

        if verbose:
            if direction is None:
                print(f"\nMove {move_count}: None")
            else:
                print(f"\nMove {move_count}: {DIRS[direction]}")
            print(board)
            print(f"Current Match-3 Score: {curr_score}")
            print(f"Projected Match-3 Score: {proj_score}")
            print(f"Q-Values: {Q_values}")
            print(f"masked: {reject_dirs}")
            if direction is None:
                print("No beneficial action found. Ending early.")
                break

        if direction is None:
            break

        board = new_board
        pos = new_pos
        # prev_dir = direction
        reject_dirs = [OPPOSITE_DIR[direction]]
        move_path.append(direction)
        move_count += 1

    final_score = find_match_3_score(board)
    if verbose:
        print("\n\nInitial Board:")
        print("Position:", initial_state[1])
        print(initial_state[0])
        print("\n\nFinal Board:")
        print(board)
        print(f"\nFinal Match-3 Score: {final_score}")
        print(move_path)

    return board, final_score
