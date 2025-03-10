import numpy as np
import torch
import random
from kubernetes_scheduler_env import KubernetesSchedulerEnv
from qmix_agent import QMIX

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
num_agents = len(env.agents)  # Always get the latest agent count
state_dim = 2
action_dim = 11

# Initialize QMIX agent
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
for episode in range(NUM_EPISODES):
    state = np.array(env.reset())
    num_agents = env.num_agents  # Update dynamically
    state = adjust_state_size(state, num_agents, state_dim)
    
    total_reward = 0
    print(f"📢 Episode {episode}: Training with {num_agents} worker nodes!")

    for step in range(MAX_STEPS):
        actions = qmix_agent.select_actions(state[:num_agents * state_dim].reshape(num_agents, state_dim), epsilon=epsilon)
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

    if episode % 1000 == 0:
        torch.save(qmix_agent.q_networks[0].state_dict(), f"qmix_model_ep{episode}.pth")

torch.save(qmix_agent.q_networks[0].state_dict(), "qmix_trained_k8s.pth")
print("✅ Training completed! Model saved as `qmix_trained_k8s.pth`.")
