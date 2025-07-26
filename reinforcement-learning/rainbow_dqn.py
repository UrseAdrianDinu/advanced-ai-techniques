import torch.nn as nn
import numpy as np
import torch
from collections import deque, namedtuple
import random
import itertools
import torch.optim as optim
import math
from dqn import get_epsilon_schedule
import gc
from torch import jit, amp

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

V_MIN = -1
V_MAX = 23
NUM_ATOMS = 51
atoms = torch.linspace(V_MIN, V_MAX, NUM_ATOMS).to("cuda")
delta_z = (V_MAX - V_MIN) / (NUM_ATOMS - 1)

def project_distribution(next_dist: torch.Tensor,
                        rewards: torch.Tensor,
                        dones: torch.Tensor,
                        gamma:float = 0.99):

    with torch.no_grad():
        batch_size = rewards.size(0)
                
        # Compute Tz (projected distribution's atoms)
        Tz = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * gamma * atoms.unsqueeze(0)
        Tz = Tz.clamp(V_MIN, V_MAX)
        
        # Compute b (mapping of Tz to nearest atom bins)
        b = (Tz - V_MIN) / delta_z
        l = b.floor().long()  # Lower bin index
        u = b.ceil().long()   # Upper bin index
        
        # Ensure l and u are within [0, NUM_ATOMS - 1]
        l = l.clamp(0, NUM_ATOMS - 1)
        u = u.clamp(0, NUM_ATOMS - 1)

        # Calculate the probability mass to be distributed
        offset = torch.arange(0, batch_size * NUM_ATOMS, NUM_ATOMS, device=rewards.device).unsqueeze(1)
        
        # Initialize projected distribution with zeros
        projected_dist = torch.zeros(batch_size, NUM_ATOMS, device=next_dist.device)

        # Distribute probability mass to nearest bins using scatter_add_
        proj_dist_l = next_dist * (u.float() - b)
        l_indices = (l + offset).view(-1)
        projected_dist.view(-1).scatter_add_(0, l_indices, proj_dist_l.view(-1))

        proj_dist_u = next_dist * (b - l.float())
        u_indices = (u + offset).view(-1)
        projected_dist.view(-1).scatter_add_(0, u_indices, proj_dist_u.view(-1))
    
    return projected_dist



class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Standard weight and bias parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        # Initialize parameters
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        # Initialize mean parameters
        mu_range = 1 / self.in_features**0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        # Initialize sigma parameters
        self.weight_sigma.data.fill_(self.sigma_init)
        self.bias_sigma.data.fill_(self.sigma_init)

    def reset_noise(self):
        # Sample from factorized Gaussian noise
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

class DuelingNoisyQNetwork(nn.Module):
    def __init__(self, input_shape, num_actions, num_atoms=51):
        super(DuelingNoisyQNetwork, self).__init__()
        self.num_atoms = num_atoms
        self.num_actions = num_actions

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)

        self.advantage = nn.Sequential(
            NoisyLinear(conv_out_size, 512),
            nn.ReLU(),
            NoisyLinear(512, num_actions * num_atoms)
        )
        self.value = nn.Sequential(
            NoisyLinear(conv_out_size, 512),
            nn.ReLU(),
            NoisyLinear(512, num_atoms)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size(0), -1)
        advantage = self.advantage(conv_out).view(-1, self.num_actions, self.num_atoms)
        value = self.value(conv_out).view(-1, 1, self.num_atoms)
        q_vals = value + advantage - advantage.mean(1, keepdim=True)
        return torch.softmax(q_vals, dim=2)

    def reset_noise(self):
        for layer in self.advantage:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
        for layer in self.value:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.size = 0
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])

    def __len__(self):
        return self.size

class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_anneal_steps=1_000_000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_end = beta_end
        self.beta_anneal_steps = beta_anneal_steps
        self.beta_increment = (beta_end - beta_start) / beta_anneal_steps
        self.epsilon = 1e-5  # Small value to avoid zero priority

    def push(self, *args):
        max_priority = np.max(self.tree.tree[-self.tree.capacity:]) if self.tree.size > 0 else 1.0
        experience = Experience(*args)
        self.tree.add(max_priority, experience)

    def sample(self, batch_size):
        batch = []
        indices = []
        priorities = []

        segment = self.tree.total() / batch_size
        self.beta = min(self.beta + self.beta_increment, self.beta_end)

        for i in range(batch_size):
            s = random.uniform(i * segment, (i + 1) * segment)
            idx, p, data = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(p)

        sampling_probs = priorities / self.tree.total()
        weights = (len(self.tree) * sampling_probs) ** (-self.beta)
        weights /= (weights.max() + 1e-5)
        weights = torch.tensor(weights, dtype=torch.float32)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (states, actions, rewards, next_states, dones, weights), indices

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.tree.update(idx, priority + self.epsilon)

    def __len__(self):
        return len(self.tree)


class RainbowDQNAgent:
    def __init__(self, input_shape, num_actions, writer, memory_capacity=200_000, gamma=0.99, lr=1e-4, n_steps=3):
        self.num_actions = num_actions
        self.gamma = gamma
        self.n_steps = n_steps
        self.writer = writer
        self.n_step_buffer = deque(maxlen=n_steps)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = torch.amp.GradScaler()
        
        # Move atoms to the correct device
        global atoms
        atoms = atoms.to(self.device)
        
        self.memory = PrioritizedReplayMemory(memory_capacity)
        self.q_network = DuelingNoisyQNetwork(input_shape, num_actions).to(self.device)
        self.q_network.train()
        self.target_network = DuelingNoisyQNetwork(input_shape, num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        
    def act(self, state, step_cnt):
        with torch.no_grad():
            if len(state.shape) == 3:
                state = state.unsqueeze(0)
            state = state.to(self.device)
            dist = self.q_network(state)
            q_values = (dist * atoms).sum(2)
            action = q_values.argmax(1).item()
        self.q_network.reset_noise()
        return action

    def act_greedy(self, state):
        with torch.no_grad():
            if len(state.shape) == 3:
                state = state.unsqueeze(0)
            state = state.to(self.device)
            dist = self.q_network(state)
            q_values = (dist * atoms).sum(2)
            return q_values.argmax(1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))

        if len(self.n_step_buffer) == self.n_steps or done:
            # Calculate multi-step reward and next state
            reward, next_state, done = self._calc_multi_step()
            self.memory.push(state, action, reward, next_state, done)
        
        if done:
            self.n_step_buffer.clear()
    
    def _calc_multi_step(self):
        # Calculate cumulative multi-step reward and next state
        reward, next_state, done = 0, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
        for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
            reward += (self.gamma ** i) * r 
            if d:
                done = d
                break
        return reward, next_state, done


    def update(self, batch_size, step_cnt, beta=0.4):
        if len(self.memory) < batch_size:
            return
        (states, actions, rewards, next_states, dones, weights), indices = self.memory.sample(batch_size)

        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        with torch.no_grad():
            next_dist_main = self.q_network(next_states)
            next_q_values_main = (next_dist_main * atoms).sum(2)
            next_actions = next_q_values_main.argmax(1)

            next_dist_target = self.target_network(next_states)
            next_dist = next_dist_target[range(batch_size), next_actions]
            projected_dist = project_distribution(next_dist, rewards, dones)

        dist = self.q_network(states)
        dist = dist.gather(1, actions.view(-1, 1, 1).expand(-1, -1, dist.size(2))).squeeze(1)

        # Compute per-sample loss for updating priorities
        per_sample_loss = -(projected_dist * torch.log(dist + 1e-8)).sum(1)
        weighted_loss = (weights * per_sample_loss).mean()

        self.optimizer.zero_grad()
        self.scaler.scale(weighted_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Update priorities using per-sample losses
        per_sample_loss_cpu = per_sample_loss.detach().cpu().numpy()
        self.memory.update_priorities(indices, per_sample_loss_cpu)
        self.q_network.reset_noise()

        if step_cnt % 5000 == 0:
            self.writer.add_scalar('Train/Loss', weighted_loss.item(), step_cnt)
            self.writer.flush()
        
        del per_sample_loss_cpu
        if step_cnt % 50_000 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        
    
    def update_target_network(self):
        """Copy weights from q_network to target_network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
