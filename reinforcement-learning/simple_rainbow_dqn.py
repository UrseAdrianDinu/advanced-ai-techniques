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
from dqn import ReplayMemory, get_epsilon_schedule

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
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions * num_atoms)
        )
        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_atoms)
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



class SimpleRainbowDQNAgent:
    def __init__(self, input_shape, num_actions, writer, memory_capacity=200_000, gamma=0.99, lr=5e-5):
        self.num_actions = num_actions
        self.gamma = gamma
        self.writer = writer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.epsilon_schedule = get_epsilon_schedule(start=1.0, end=0.1, steps=700_000)
        # Move atoms to the correct device
        global atoms
        atoms = atoms.to(self.device)
        
        self.memory = ReplayMemory(memory_capacity)
        self.q_network = DuelingNoisyQNetwork(input_shape, num_actions).to(self.device)
        self.q_network.train()
        self.target_network = DuelingNoisyQNetwork(input_shape, num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        
    def act(self, state, step_cnt):
        epsilon = next(self.epsilon_schedule)

        if random.random() < epsilon:
            action = torch.randint(self.num_actions, (1,)).item()
        else:
            with torch.no_grad():
                if len(state.shape) == 3:
                    state = state.unsqueeze(0)
                state = state.to(self.device)
                dist = self.q_network(state)
                q_values = (dist * atoms).sum(2)
                action = q_values.argmax(1).item()
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
        self.memory.push(state.detach(), action, reward, next_state.detach(), done)

    def update(self, batch_size, step_cnt, beta=0.4):
        if len(self.memory) < batch_size:
            return
        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        with torch.no_grad():
            next_dist_main = self.q_network(next_states)
            next_q_values_main = (next_dist_main * atoms).sum(2)
            next_actions = next_q_values_main.argmax(1)

            next_dist_target = self.target_network(next_states)
            next_dist = next_dist_target[range(batch_size), next_actions]
            projected_dist = project_distribution(next_dist, rewards, dones)

        dist = self.q_network(states)
        dist = dist.gather(1, actions.view(-1, 1, 1).expand(-1, -1, dist.size(2))).squeeze(1)

        loss = -torch.sum(projected_dist * torch.log(dist + 1e-8), dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if step_cnt % 5000 == 0:
            self.writer.add_scalar('Train/Loss', loss.item(), step_cnt)
            self.writer.flush()
    
    def update_target_network(self):
        """Copy weights from q_network to target_network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
