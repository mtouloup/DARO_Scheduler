import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class QNetwork(nn.Module):
    """Individual Q-Network for each agent"""
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class QMIX:
    """QMIX algorithm for multi-agent learning"""
    def __init__(self, num_agents, input_dim, output_dim, lr=1e-3, gamma=0.99):
        self.num_agents = num_agents
        self.q_networks = [QNetwork(input_dim, output_dim) for _ in range(num_agents)]
        self.optimizer = optim.Adam(self.get_parameters(), lr=lr)
        self.gamma = gamma
        self.criterion = nn.MSELoss()

    def get_parameters(self):
        """Returns parameters for all networks"""
        params = []
        for q_net in self.q_networks:
            params += list(q_net.parameters())
        return params

    def select_actions(self, states, epsilon=0.1):
        """Epsilon-greedy action selection for each agent, dynamically adjusting to agent count."""
        num_agents = len(states)  # Get the current number of agents

        # Ensure `self.q_networks` matches the number of agents
        if len(self.q_networks) < num_agents:
            missing_agents = num_agents - len(self.q_networks)
            for _ in range(missing_agents):
                self.q_networks.append(QNetwork(states.shape[1], 11))  # Initialize new Q-networks as needed

        actions = []
        for i in range(num_agents):
            state_tensor = torch.tensor(states[i], dtype=torch.float32).unsqueeze(0)
            if np.random.rand() < epsilon:
                action = np.random.randint(1, 10)  # Random bid
            else:
                with torch.no_grad():
                    q_values = self.q_networks[i](state_tensor)
                    action = torch.argmax(q_values).item()
            actions.append(action)
        return actions


    def train(self, experiences):
        """Train QMIX with batch experience while handling dynamic agent count."""
        states, actions, rewards, next_states = zip(*experiences)

        # Find the maximum number of agents in this batch
        max_agents = max(len(state) for state in states)

        # Ensure we do not index beyond available Q-networks
        max_agents = min(max_agents, len(self.q_networks))

        # Pad or truncate states, next_states, rewards, and actions to match `max_agents`
        def pad_or_truncate(data, target_size, feature_dim=None):
            padded_data = []
            for item in data:
                if len(item) > target_size:
                    padded_data.append(item[:target_size])  # Truncate
                else:
                    pad_size = target_size - len(item)
                    if feature_dim:  # For states and next_states
                        pad_array = np.zeros((pad_size, feature_dim), dtype=np.float32)
                    else:  # For rewards and actions
                        pad_array = [0] * pad_size
                    padded_data.append(np.vstack((item, pad_array)) if feature_dim else item + pad_array)
            return np.array(padded_data)

        # Ensure consistent shapes
        states = pad_or_truncate(states, max_agents, feature_dim=len(states[0][0]))
        next_states = pad_or_truncate(next_states, max_agents, feature_dim=len(next_states[0][0]))
        rewards = pad_or_truncate(rewards, max_agents)
        actions = pad_or_truncate(actions, max_agents)

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)

        loss = 0
        for i in range(max_agents):  # Ensure `i` does not exceed available Q-networks
            q_values = self.q_networks[i](states[:, i, :])
            next_q_values = self.q_networks[i](next_states[:, i, :]).max(1)[0]
            target_q_values = rewards[:, i] + self.gamma * next_q_values

            loss += self.criterion(q_values.gather(1, actions[:, i].unsqueeze(1)).squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

