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

    def get_state_vector(self, task):
        return np.array([
            self.cpu_capacity,
            self.memory_capacity,
            self.cpu_capacity - self.current_cpu_usage,
            self.memory_capacity - self.current_memory_usage,
            task.cpu_request,
            task.memory_request
        ], dtype=np.float32)


class Broker:
    def __init__(self, agents):
        self.agents = agents

    def select_actions_from_qmix(self, qmix_agent, state, epsilon=0.1):
        return qmix_agent.select_actions(state, epsilon)

class KubernetesSchedulerEnv(gym.Env):
    def __init__(self, min_agents=3, max_agents=100):
        super(KubernetesSchedulerEnv, self).__init__()
        self.min_agents = min_agents
        self.max_agents = max_agents
        self.num_agents = random.randint(self.min_agents, self.max_agents)
        self.agents = self._create_agents(self.num_agents)
        self.broker = Broker(self.agents)
        self.action_space = spaces.MultiDiscrete([11] * self.num_agents)
        self.observation_space = spaces.Box(low=0, high=256, shape=(self.max_agents, 6), dtype=np.float32)
        self.current_task = None

    def _create_agents(self, num_agents):
        return [Agent(f"node-{i}", random.randint(4, 64), random.randint(8, 256)) for i in range(num_agents)]

    def _generate_task(self):
        return Task(
            task_id=random.randint(1, 1000),
            cpu_request=random.uniform(0.5, 16),
            memory_request=random.uniform(1, 128)
        )

    def set_current_task(self, task):
        self.current_task = task

    def reset(self):
        self.num_agents = random.randint(self.min_agents, self.max_agents)
        print(f"🔄 Reset: {self.num_agents} agents initialized")
        self.agents = self._create_agents(self.num_agents)
        self.broker = Broker(self.agents)
        self.current_task = None
        return None

    def evaluate_bids(self, actions):
        task = self.current_task
        valid_bids = []

        print(f"\n📌 Task {task.task_id}: CPU={task.cpu_request:.2f}, Mem={task.memory_request:.2f}")

        for i in range(self.num_agents):
            agent = self.agents[i]
            bid = actions[i]

            available_cpu = agent.cpu_capacity - agent.current_cpu_usage
            available_memory = agent.memory_capacity - agent.current_memory_usage
           
            if bid > 0 and task.cpu_request <= available_cpu and task.memory_request <= available_memory:
                valid_bids.append((bid, agent))
            else:
                if bid >= 0:
                    print(f"   ❌ Agent {agent.node_id} bid {bid} but lacks resources "
                        f"(CPU: {available_cpu:.2f}/{task.cpu_request:.2f}, "
                        f"Memory: {available_memory:.2f}/{task.memory_request:.2f})")

        print(f"\n📢 New Task {task.task_id}: Requires CPU={task.cpu_request:.2f}, Memory={task.memory_request:.2f}")
        for i in range(self.num_agents):
            agent = self.agents[i]
            print(f"   🔹 Agent {agent.node_id}: Available CPU={agent.cpu_capacity - agent.current_cpu_usage:.2f}, "
                f"Memory={agent.memory_capacity - agent.current_memory_usage:.2f}, Bid={actions[i]}")

        if valid_bids:
            valid_bids.sort(reverse=True, key=lambda x: x[0])
            highest_bid, winner_agent = valid_bids[0]
            winner_index = self.agents.index(winner_agent)

            # Apply task assignment
            winner_agent.current_cpu_usage += task.cpu_request
            winner_agent.current_memory_usage += task.memory_request

            print(f"🏆 Task {task.task_id} assigned to {winner_agent.node_id} with highest bid {highest_bid}")
            print(f"   ✅ New Usage - CPU: {winner_agent.current_cpu_usage:.2f}/{winner_agent.cpu_capacity}, "
                f"Memory: {winner_agent.current_memory_usage:.2f}/{winner_agent.memory_capacity}")
        else:
            winner_index = -1

        return winner_index

    def calculate_rewards(self, winner_index):
        rewards = [-0.1] * self.num_agents
        if winner_index != -1:
            rewards[winner_index] = 1.0
        return rewards

    def step(self, actions):
        winner_index = self.evaluate_bids(actions)
        rewards = self.calculate_rewards(winner_index)
        return self.get_obs(), rewards, False, {}


    def get_obs(self):
        return np.array([
            agent.get_state_vector(self.current_task) for agent in self.agents
        ])
    
    