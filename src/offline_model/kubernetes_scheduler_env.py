import numpy as np
import random
import gym
from gym import spaces

class Task:
    """Represents a Kubernetes Pod Deployment Request"""
    def __init__(self, task_id, cpu_request, memory_request):
        self.task_id = task_id
        self.cpu_request = cpu_request
        self.memory_request = memory_request

class Agent:
    """Represents a Kubernetes Worker Node"""
    def __init__(self, node_id, cpu_capacity, memory_capacity):
        self.node_id = node_id
        self.cpu_capacity = cpu_capacity
        self.memory_capacity = memory_capacity
        self.current_cpu_usage = 0.0
        self.current_memory_usage = 0.0

    def get_state_vector(self):
        """Returns the remaining CPU and memory capacity as a state vector"""
        return np.array([
            self.cpu_capacity - self.current_cpu_usage,
            self.memory_capacity - self.current_memory_usage
        ], dtype=np.float32)

    def calculate_bid(self, task):
        """Agents generate a random bid between 1 and 10 if they have enough resources"""
        available_cpu = self.cpu_capacity - self.current_cpu_usage
        available_memory = self.memory_capacity - self.current_memory_usage

        # Print debug info
        print(f"🖥️ Agent {self.node_id} - Available: CPU={available_cpu:.2f}, Memory={available_memory:.2f}")
        print(f"   📌 Task {task.task_id} - Requires: CPU={task.cpu_request:.2f}, Memory={task.memory_request:.2f}")

        if task.cpu_request <= available_cpu and task.memory_request <= available_memory:
            bid = np.random.randint(1, 11)  # Random bid from 1 to 10
            print(f"   ✅ Agent {self.node_id} bids {bid}")
            return bid
        else:
            print(f"   ❌ Agent {self.node_id} cannot bid (Insufficient resources)")
            return -1  # No bid if insufficient resources

class Broker:
    """Broker that collects bids but does NOT decide—decision is learned via MARL"""
    def __init__(self, agents):
        self.agents = agents

    
    def collect_bids(self, task):
        """Collect valid bids from agents and return a list of (bid_value, agent) tuples"""
        bids = []
        print(f"\n📢 New Task {task.task_id}: Requires CPU={task.cpu_request:.2f}, Memory={task.memory_request:.2f}")

        for i in range(len(self.agents)):
            agent = self.agents[i]
            bid = agent.calculate_bid(task)  # Call `calculate_bid()`

            print(f"   🔹 Agent {agent.node_id}: Bid={bid}")  # Print agent bid
            if bid != -1:
                bids.append((bid, agent))  # Store as (bid_value, agent) tuple

        return bids  # Always return a list

class KubernetesSchedulerEnv(gym.Env):
    """Multi-Agent Reinforcement Learning (MARL) Kubernetes Scheduling Environment"""
    def __init__(self, min_agents=3, max_agents=100):
        super(KubernetesSchedulerEnv, self).__init__()
        self.min_agents = min_agents
        self.max_agents = max_agents
        self.num_agents = random.randint(self.min_agents, self.max_agents)
        self.agents = self._create_agents(self.num_agents)
        self.broker = Broker(self.agents)  # Broker added back!
        self.action_space = spaces.MultiDiscrete([11] * self.num_agents)  # Each agent submits a bid (0-10)
        self.observation_space = spaces.Box(low=0, high=256, shape=(self.max_agents, 2), dtype=np.float32)
        self.current_task = None

    def _create_agents(self, num_agents):
        return [Agent(f"node-{i}", random.randint(4, 64), random.randint(8, 256)) for i in range(num_agents)]

    def reset(self):
        """Resets environment and returns initial observations"""
        self.num_agents = random.randint(self.min_agents, self.max_agents)
        self.agents = self._create_agents(self.num_agents)
        self.broker = Broker(self.agents)  # Reinitialize broker
        self.current_task = Task(
            task_id=random.randint(1, 1000),
            cpu_request=random.uniform(0.5, 16),
            memory_request=random.uniform(1, 128)
        )
        return self.get_obs()

    def step(self):
        """Processes agent bids and assigns the task"""
        valid_bids = self.broker.collect_bids(self.current_task)  # Collect bids

        if valid_bids:
            # Sort by bid value in descending order
            valid_bids.sort(reverse=True, key=lambda x: x[0])
            highest_bid, winner_agent = valid_bids[0]  # Select the agent with the highest bid
            
            print(f"🏆 Task {self.current_task.task_id} assigned to {winner_agent.node_id} with highest bid {highest_bid}")

            # Find the index of the winner agent
            winner_index = self.agents.index(winner_agent)
        else:
            print("❌ No agent could handle the task. Skipping task assignment.")
            winner_agent = None
            winner_index = -1

        # Reward system
        rewards = [-0.1] * self.num_agents  # Default negative reward for losing agents
        if winner_agent:
            rewards[winner_index] = 1  # Assign positive reward to winner

        # Generate a new task
        self.current_task = Task(
            task_id=random.randint(1, 1000),
            cpu_request=random.uniform(0.5, 16),
            memory_request=random.uniform(1, 128)
        )

        return self.get_obs(), rewards, False, {}

    def get_obs(self):
        """Returns the state of all agents"""
        return np.array([agent.get_state_vector() for agent in self.agents])
