import torch
import numpy as np
import random
from kubernetes_scheduler_env import KubernetesSchedulerEnv, Task
from qmix_agent import QMIX

# Load environment
env = KubernetesSchedulerEnv(min_agents=3, max_agents=10)
num_agents = env.num_agents
state_dim = 2
action_dim = 11

# Load trained model
trained_model_path = "qmix_general_model.pth"
checkpoint = torch.load(trained_model_path)

# Initialize QMIX agent
qmix_agent = QMIX(num_agents=num_agents, input_dim=state_dim, output_dim=action_dim)
for i in range(len(qmix_agent.q_networks)):
    qmix_agent.q_networks[i].load_state_dict(checkpoint)
    qmix_agent.q_networks[i].eval()

def adjust_state_size(state, num_agents, state_dim):
    state = np.array(state)
    if state.shape[0] > num_agents:
        state = state[:num_agents]
    elif state.shape[0] < num_agents:
        pad_size = num_agents - state.shape[0]
        pad_array = np.zeros((pad_size, state_dim), dtype=np.float32)
        state = np.vstack((state, pad_array))
    return state

# Testing loop
num_test_episodes = 1
max_steps = 50
total_rewards = []

print("\n📢 **Evaluating Trained QMIX Model on Kubernetes Scheduling**")

for episode in range(num_test_episodes):
    state = np.array(env.reset())
    num_agents = env.num_agents
    state = adjust_state_size(state, num_agents, state_dim)
    episode_reward = 0

    print(f"\n🚀 Episode {episode + 1}: Simulating Kubernetes Pod Scheduling")

    for step in range(max_steps):
        # Generate and set a new task for this step
        task = Task(
            task_id=random.randint(1, 1000),
            cpu_request=random.uniform(0.5, 16),
            memory_request=random.uniform(1, 128)
        )
        env.set_current_task(task)

        print(f"\n📌 Task {task.task_id}: CPU={task.cpu_request:.2f}, Mem={task.memory_request:.2f}")

        # Get greedy actions (epsilon = 0)
        actions = env.broker.select_actions_from_qmix(
            qmix_agent,
            state[:num_agents * state_dim].reshape(num_agents, state_dim),
            epsilon=0
        )

        print(f"🎯 Bidding Results:")
        for i, bid in enumerate(actions):
            print(f"   🤖 Agent {i} bids {bid}")

        # Step into environment
        next_state, rewards, _, _ = env.step(actions)
        next_state = adjust_state_size(np.array(next_state), num_agents, state_dim)

        if max(rewards) > 0:
            winning_agent = rewards.index(max(rewards))
            print(f"🏆 Task assigned to Agent {winning_agent} with bid {actions[winning_agent]}")
        else:
            print("❌ No valid bids, task was skipped.")

        episode_reward += sum(rewards)
        state = next_state

    total_rewards.append(episode_reward)
    print(f"\n🏆 Episode {episode + 1}: Total Reward = {episode_reward}")

# Evaluation Summary
avg_reward = np.mean(total_rewards)
print("\n📊 **Evaluation Summary**")
print(f"🔥 Average Total Reward per Episode: {avg_reward}")
print("✅ Testing completed!")
