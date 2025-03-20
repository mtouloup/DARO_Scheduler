import numpy as np
import torch
import random
import pandas as pd
import matplotlib.pyplot as plt
from kubernetes_scheduler_env import KubernetesSchedulerEnv
from qmix_agent import QMIX, QNetwork

# Hyperparameters
NUM_EPISODES = 100
MAX_STEPS = 50
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 10000
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.999
LEARNING_RATE = 1e-3

# Initialize Kubernetes environment
env = KubernetesSchedulerEnv(min_agents=3, max_agents=10)
state_dim = 2
action_dim = 11

# Initialize QMIX agent
num_agents = env.num_agents  # Get initial number of agents
qmix_agent = QMIX(num_agents=num_agents, input_dim=state_dim, output_dim=action_dim, lr=LEARNING_RATE, gamma=GAMMA)

# Replay memory
replay_buffer = []

def adjust_state_size(state, num_agents, state_dim):
    """Ensure state is correctly formatted as (num_agents, state_dim)."""
    state = np.array(state)

    if state.shape[0] > num_agents:
        state = state[:num_agents]
    elif state.shape[0] < num_agents:
        pad_size = num_agents - state.shape[0]
        pad_array = np.zeros((pad_size, state_dim), dtype=np.float32)
        state = np.vstack((state, pad_array))

    return state

# Training loop
epsilon = EPSILON_START
bid_tracking = []  # ✅ Fixed bid tracking

for episode in range(NUM_EPISODES):
    state = np.array(env.reset())
    num_agents = env.num_agents  # Dynamically update the number of agents
    state = adjust_state_size(state, num_agents, state_dim)

    total_reward = 0
    print(f"📢 Episode {episode}: Training with {num_agents} worker nodes!")

    # ✅ Ensure bid tracking is correctly sized for this episode
    episode_bids = []  # Store all bids per step

    for step in range(MAX_STEPS):
        actions = qmix_agent.select_actions(state[:num_agents * state_dim].reshape(num_agents, state_dim), epsilon=epsilon)

        # ✅ Store all bids step by step
        episode_bids.append([int(bid) for bid in actions]) 

        next_state, rewards, _, _ = env.step(actions)
        next_state = adjust_state_size(np.array(next_state), num_agents, state_dim)
        replay_buffer.append((state, actions, rewards, next_state))

        if len(replay_buffer) > REPLAY_BUFFER_SIZE:
            replay_buffer.pop(0)

        if len(replay_buffer) >= BATCH_SIZE:
            batch = random.sample(replay_buffer, BATCH_SIZE)
            qmix_agent.train(batch)

        state = next_state
        total_reward += sum(rewards)

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    print(f"🏆 Episode {episode}: Total Reward = {total_reward}")

    # ✅ Append episode bids properly
    bid_tracking.append(episode_bids)

# ✅ Convert bid tracking into DataFrame
# Ensure all episodes are stored correctly and properly padded
max_agents = max(len(max(episode, key=len)) for episode in bid_tracking)  # Find max agents across episodes

# Flatten and format the bid tracking data properly
formatted_bids = []
for episode in bid_tracking:
    episode_bids = []
    for step in episode:
        # Ensure all bids are integers
        int_bids = [int(bid) for bid in step]
        episode_bids.append(int_bids + [None] * (max_agents - len(int_bids)))  # Use None instead of NaN
    formatted_bids.extend(episode_bids)

# Convert to a DataFrame
bid_data = pd.DataFrame(formatted_bids)

# Convert all numeric columns explicitly to integers
bid_data = bid_data.astype("Int64")  # Uses Pandas nullable integer type to handle missing values

# Save the corrected bid data
bid_data.to_csv("agent_bids.csv", index=False)

# ✅ Aggregating all agent models into one general model
print("\n📢 Aggregating agent models into a single general model...")

# Initialize a new general model
general_model = QNetwork(state_dim, action_dim)
general_state_dict = general_model.state_dict()

# Average the weights across all trained agents directly from memory
for key in general_state_dict.keys():
    general_state_dict[key] = sum(qmix_agent.q_networks[i].state_dict()[key] for i in range(num_agents)) / num_agents

# Load the averaged weights into the general model
general_model.load_state_dict(general_state_dict)

# Save only the general model for deployment
torch.save(general_model.state_dict(), "qmix_general_model.pth")

print("✅ Training completed! General model saved as `qmix_general_model.pth`.")
