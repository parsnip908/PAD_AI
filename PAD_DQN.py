import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
import numpy as np

import board_utils

# Constants
BATCH_SIZE = 32
GAMMA = 0.99
LR = 1e-3
BUFFER_SIZE = 10000
TARGET_UPDATE_FREQ = 100
NUM_ACTIONS = 4

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using cuda") if torch.cuda.is_available() else print("using cpu")


class PAD_DQN(nn.Module):
    def __init__(self):
        super(PAD_DQN, self).__init__()

        # Convolutional layers for 5x6x8 input
        self.conv3_1 = nn.Conv2d(7, 32, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(7, 16, kernel_size=4, padding=0)
        self.conv5 = nn.Conv2d(7, 16, kernel_size=5, padding=0)
        self.convB = nn.Conv2d(7, 32, kernel_size=(5,6), padding=0)

        self.flat_size = 64*5*6 + 16*2*3 + 16*1*2 + 32

        # Timer input (scalar) is concatenated with conv output
        self.fc_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.flat_size + 1, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4)  # 5 actions: up/down/left/right/finish
        )

    def forward(self, board, position, game_values):

        # print("forward")
        batch_size = board.size()[0]
        board_tensor = F.one_hot(board, num_classes=6).permute(0, 3, 1, 2)
        # loc_tensor = torch.zeros(batch_size,1,5,6).to(device)

        # for i in range(batch_size):
        #     loc_tensor[i][0][game_values[i][0]][game_values[i][1]] = 1

        # print(board_tensor.size())
        # print(loc_tensor.size())
        image_input = torch.cat([position.unsqueeze(1), board_tensor], 1).float()

        x3 = self.conv3_1(image_input)
        x3 = F.relu(x3)
        x3 = self.conv3_2(x3)
        x3 = x3.view(-1, 64*5*6)

        x4 = self.conv4(image_input)
        x5 = self.conv5(image_input)
        xB = self.convB(image_input)

        x4 = x4.view(-1, 16*2*3)
        x5 = x5.view(-1, 16*1*2)
        xB = xB.view(-1, 32)

        # Concatenate game_values input
        x = torch.cat([x3, x4, x5, xB, (game_values[:,2:3].float()/100.0)], 1)

        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        q_values = self.fc_layers(x)  # Output shape: (B, 5)
        return q_values


old_match_score = 0
old_prox_score = 0

def init_score(board):
    global old_match_score
    global old_prox_score
    old_match_score = board_utils.find_match_3_score(board)
    old_prox_score = board_utils.proximity_score(board)

def compute_reward(state, action, next_state, done):
    """Compute reward for taking action in current state."""

    global old_match_score
    global old_prox_score
    # old_match_score = board_utils.find_match_3_score(state[0])
    new_match_score = board_utils.find_match_3_score(next_state[0])
    # old_prox_score = board_utils.proximity_score(state[0])
    new_prox_score = board_utils.proximity_score(next_state[0])

    reward = new_match_score - old_match_score
    if new_prox_score > old_prox_score:
        reward += new_prox_score - old_prox_score

    old_match_score = new_match_score
    old_prox_score = new_prox_score


    if action == 4:
        reward -= 5
    elif torch.equal(state[2], next_state[2]):
        reward -= 20
    
    return reward

def get_next_state(state, action):
    """Compute next state given current state and action."""
    # print(state)
    # print(action)

    row, col = int(state[2][0].item()), int(state[2][1].item())
    swap_row, swap_col = row, col

    if action == 0:       # up
        swap_row -= 1
    elif action == 1:     # down
        swap_row += 1
    elif action == 2:     # left
        swap_col -= 1
    elif action == 3:     # right
        swap_col += 1
    else:
        return state  # Invalid direction

    if 0 <= swap_row < 5 and 0 <= swap_col < 6:
        with torch.no_grad():
            board = state[0].clone()
            temp = board[row, col].item()
            board[row, col] = board[swap_row, swap_col]
            board[swap_row, swap_col] = temp

            pos = state[1].clone()
            pos[row, col] = 0
            pos[swap_row, swap_col] = 1
            
            game = state[2].clone()
            game[0] = swap_row
            game[1] = swap_col
            game[2] -= 1

        # print(board)
        # print(game)
        return (board, pos, game)
    return state

def is_done(game, action):
    if game[2] == 0 or action == 4:
        return True
    return False

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, BATCH_SIZE):
        batch = random.sample(self.buffer, BATCH_SIZE)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# Training step
def train_step():
    if len(replay_buffer) < BATCH_SIZE:
        return

    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    # Convert to tensors
    boards = torch.stack([s[0] for s in states]).to(device)      # (B, 5, 6)
    positions = torch.stack([s[1] for s in states]).to(device)  # (B, 3)
    game_values = torch.stack([s[2] for s in states]).to(device)  # (B, 3)
    # print(game_values)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)  # (B, 1)
    rewards = torch.tensor(rewards, dtype=torch.float).unsqueeze(1).to(device)  # (B, 1)
    dones = torch.tensor(dones, dtype=torch.bool).unsqueeze(1).to(device)

    next_boards = torch.stack([s[0] for s in next_states]).to(device)
    next_positions = torch.stack([s[1] for s in next_states]).to(device)
    next_game_values = torch.stack([s[2] for s in next_states]).to(device)

    # Current Q values
    q_values = policy_net(boards, positions, game_values).gather(1, actions)

    # Target Q values
    with torch.no_grad():
        next_q_values = target_net(next_boards, next_positions, next_game_values).max(1, keepdim=True)[0]
        target_q_values = rewards + (GAMMA * next_q_values * (~dones))

    # Loss and optimization
    loss = F.mse_loss(q_values, target_q_values)

    # print(q_values)
    # print(target_q_values)


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    model = PAD_DQN()

    # Dummy batch input
    # torch.manual_seed(3)
    board = torch.randint(0, 6, (BATCH_SIZE, 5, 6))      # board state
    player_y = torch.randint(0, 5, (BATCH_SIZE, 1))
    player_x = torch.randint(0, 6, (BATCH_SIZE, 1))
    game_time = torch.randint(90, 100, (BATCH_SIZE, 1))  # scalar timer
    game = torch.cat([player_y, player_x, game_time], 1)

    position = torch.zeros(BATCH_SIZE,5,6)

    for i in range(BATCH_SIZE):
        position[i][game[i][0]][game[i][1]] = 1


    print(board)
    print(game)

    # for b in board:
    #     print(board_utils.proximity_score(b))

    q_values = model(board, position, game)  # shape: (32, 5)
    print(q_values)

    # exit()

    # Initialize models and optimizer
    policy_net = PAD_DQN().to(device)
    target_net = PAD_DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    # Main loop
    for episode in range(15000):  # number of episodes
        # Initialize state
        board = torch.randint(0, 6, (5, 6))
        player_y = torch.randint(0, 5, (1,))
        player_x = torch.randint(0, 6, (1,))
        timer = torch.randint(90, 100, (1,))
        game = torch.cat([player_y, player_x, timer], 0)
        
        position = torch.zeros(5,6)
        position[game[0]][game[1]] = 1

        init_score(board)
        state = (board, position, game)

        done = False
        while not done:
            # ε-greedy action selection
            epsilon = max(0.10, 1.0 - episode / 8000)
            if random.random() < epsilon:
                action = random.randint(0, NUM_ACTIONS-1)
            else:
                with torch.no_grad():
                    q_vals = policy_net(
                        state[0].unsqueeze(0).to(device), #board
                        state[1].unsqueeze(0).to(device), #position
                        state[2].unsqueeze(0).to(device)  #game
                    )
                    action = q_vals.argmax().item()

            # Apply action
            
            next_state = get_next_state(state, action)
            done = is_done(state[2], action)
            reward = compute_reward(state, action, next_state, done)

            # Save to replay buffer
            replay_buffer.push(state, action, reward, next_state, done)

            if episode % TARGET_UPDATE_FREQ == 0:
                print(state[0], state[1].int64(), state[2], action, reward, sep='\n')
                print('-')
            # Advance state
            state = next_state

            # Train
            train_step()
        if episode % TARGET_UPDATE_FREQ == 0:
            print(len(replay_buffer))
            print(f"episode {episode} / 10000")
            print("----------------")

        # Update target network
        if episode % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())
