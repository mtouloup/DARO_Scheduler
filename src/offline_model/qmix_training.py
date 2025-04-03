import numpy as np
import torch
import random
import pandas as pd
import time
from kubernetes_scheduler_env import KubernetesSchedulerEnv, Task
from qmix_agent import QMIX, QNetwork

# Hyperparameters
NUM_EPISODES = 1
MAX_STEPS = 10
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 10000
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.999
LEARNING_RATE = 1e-3

# Initialize Kubernetes environment
env = KubernetesSchedulerEnv(min_agents=3, max_agents=10)
state_dim = 6
action_dim = 11

# Initialize QMIX agent
num_agents = env.num_agents
qmix_agent = QMIX(num_agents=num_agents, input_dim=state_dim, output_dim=action_dim, lr=LEARNING_RATE, gamma=GAMMA)

# Replay memory
replay_buffer = []

# Decision log for future reward analysis
decision_log = []

def adjust_state_size(state, num_agents, state_dim):
    state = np.array(state)
    if state.shape[0] > num_agents:
        state = state[:num_agents]
    elif state.shape[0] < num_agents:
        pad_size = num_agents - state.shape[0]
        pad_array = np.zeros((pad_size, state_dim), dtype=np.float32)
        state = np.vstack((state, pad_array))
    return state

epsilon = EPSILON_START
bid_tracking = []

for episode in range(NUM_EPISODES):
    env.reset()
    num_agents = env.num_agents

    first_task = Task(
        task_id=random.randint(1, 1000),
        cpu_request=random.uniform(0.5, 16),
        memory_request=random.uniform(1, 128)
    )
    env.set_current_task(first_task)

    state = adjust_state_size(env.get_obs(), num_agents, state_dim)
    total_reward = 0

    print(f"\n📲 Episode {episode}: Training with {num_agents} worker nodes!")

    for i, agent in enumerate(env.agents):
        cpu_remaining = agent.cpu_capacity - agent.current_cpu_usage
        mem_remaining = agent.memory_capacity - agent.current_memory_usage
        #print(f"   🖥️ Agent {i}: CPU={cpu_remaining:.2f}, Mem={mem_remaining:.2f}")

    episode_bids = []

    for step in range(MAX_STEPS):
        # ✅ Generate task and set it explicitly
        task = Task(
            task_id=random.randint(1, 1000),
            cpu_request=random.uniform(0.5, 16),
            memory_request=random.uniform(1, 128)
        )
        env.set_current_task(task)

        print(f"\n📌 Task {task.task_id}: CPU={task.cpu_request:.2f}, Mem={task.memory_request:.2f}")

        actions = env.broker.select_actions_from_qmix(
            qmix_agent,
            state[:num_agents * state_dim].reshape(num_agents, state_dim),
            epsilon=epsilon
        )

        
        for i, bid in enumerate(actions):

            decision_log.append({
                "episode": episode,
                "step": step,
                "timestamp": time.time(),
                "task_id": task.task_id,
                "agent_id": i,
                "state": state[i].tolist(),
                "action": int(bid)
            })

        episode_bids.append([int(bid) for bid in actions])
        next_state, rewards, _, _ = env.step(actions)
        next_state = adjust_state_size(np.array(next_state), num_agents, state_dim)

        if max(rewards) > 0:
            winning_agent = rewards.index(max(rewards))
            print(f"🏆 Task assigned to Agent {winning_agent} with bid {actions[winning_agent]}")
        else:
            print("❌ No valid bids, task was skipped.")

        replay_buffer.append((state, actions, rewards, next_state))

        if len(replay_buffer) > REPLAY_BUFFER_SIZE:
            replay_buffer.pop(0)

        if len(replay_buffer) >= BATCH_SIZE:
            batch = random.sample(replay_buffer, BATCH_SIZE)
            qmix_agent.train(batch)

        state = next_state
        total_reward += sum(rewards)

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    print(f"\n🏆 Episode {episode}: Total Reward = {total_reward}")
    bid_tracking.append(episode_bids)

    print("🔄 Updating general model from agent models...")
    general_model = QNetwork(state_dim, action_dim)
    general_state_dict = general_model.state_dict()
    for key in general_state_dict:
        general_state_dict[key] = sum(qmix_agent.q_networks[i].state_dict()[key] for i in range(num_agents)) / num_agents
    general_model.load_state_dict(general_state_dict)

    for i in range(num_agents):
        qmix_agent.q_networks[i].load_state_dict(general_state_dict)
    print("✅ Agent models synced with general model.")

# Save bid data
max_agents = max(len(max(episode, key=len)) for episode in bid_tracking)
formatted_bids = []
for episode in bid_tracking:
    for step in episode:
        int_bids = [int(bid) for bid in step]
        formatted_bids.append(int_bids + [None] * (max_agents - len(int_bids)))
bid_data = pd.DataFrame(formatted_bids)
bid_data.to_csv("agent_bids.csv", index=False)

# Save decision log
pd.DataFrame(decision_log).to_csv("decision_log.csv", index=False)

# Save model
torch.save(general_model.state_dict(), "qmix_general_model.pth")
print("✅ Final general model saved as `qmix_general_model.pth`.")
