import torch
import numpy as np
from kubernetes_scheduler_env import KubernetesSchedulerEnv
from qmix_agent import QMIX

# Load environment
env = KubernetesSchedulerEnv(min_agents=3, max_agents=10)
num_agents = len(env.agents)  # Get the actual number of agents
state_dim = 2  # Each agent has (CPU capacity, Memory capacity)
action_dim = 11  # Each agent can bid between 0-10

# Load trained model
trained_model_path = "qmix_trained_k8s.pth"
checkpoint = torch.load(trained_model_path)

# Initialize QMIX agent with correct number of agents
qmix_agent = QMIX(num_agents=num_agents, input_dim=state_dim, output_dim=action_dim)
for i in range(len(qmix_agent.q_networks)):  # Load trained networks
    qmix_agent.q_networks[i].load_state_dict(checkpoint)

qmix_agent.q_networks = [net.eval() for net in qmix_agent.q_networks]  # Set models to evaluation mode

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

# Start Testing
num_test_episodes = 1
total_rewards = []

print("\n📢 **Evaluating Trained QMIX Model on Kubernetes Scheduling**")

for episode in range(num_test_episodes):
    state = np.array(env.reset())
    num_agents = len(env.agents)  # Always get the latest number of agents
    state = adjust_state_size(state, num_agents, state_dim)  # Ensure correct shape

    episode_reward = 0
    print(f"\n🚀 Episode {episode + 1}: Simulating Kubernetes Pod Scheduling")

    for step in range(50):  # Simulate 50 task deployments
        actions = qmix_agent.select_actions(state.reshape(num_agents, state_dim), epsilon=0)  # Fully greedy (best action)
        
        next_state, rewards, _, _ = env.step(actions)
        next_state = adjust_state_size(np.array(next_state), num_agents, state_dim)

        episode_reward += sum(rewards)
        state = next_state

        assigned_node = f"node-{actions.index(max(actions))}"

    total_rewards.append(episode_reward)
    print(f"🏆 Episode {episode + 1}: Total Reward = {episode_reward}")

# Evaluation Summary
avg_reward = np.mean(total_rewards)
print("\n📊 **Evaluation Summary**")
print(f"🔥 Average Total Reward per Episode: {avg_reward}")
print("✅ Testing completed!")
