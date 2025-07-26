
import torch.nn as nn
import numpy as np
import torch
from collections import deque, namedtuple
import random
import itertools
import torch.optim as optim
from dqn import ReplayMemory, get_epsilon_schedule

V_MIN = -1
V_MAX = 23
NUM_ATOMS = 51
atoms = torch.linspace(V_MIN, V_MAX, NUM_ATOMS).cuda()  
delta_z = (V_MAX - V_MIN) / (NUM_ATOMS - 1)


def project_distribution(next_dist, rewards, dones, gamma=0.99):
    batch_size = rewards.size(0)
    
    # Define atom values
    support = torch.linspace(V_MIN, V_MAX, NUM_ATOMS).to(rewards.device)  # Atom values
    
    # Compute Tz (projected distribution's support)
    Tz = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * gamma * support.unsqueeze(0)
    Tz = Tz.clamp(V_MIN, V_MAX)
    
    # Compute b (mapping of Tz to nearest atom bins)
    b = (Tz - V_MIN) / delta_z
    l = b.floor().long()  # Lower bin index
    u = b.ceil().long()   # Upper bin index

    # Offset used for indexing
    offset = torch.arange(0, batch_size * NUM_ATOMS, NUM_ATOMS, device=rewards.device).unsqueeze(1)
    
    # Initialize projected distribution with zeros
    projected_dist = torch.zeros(batch_size, NUM_ATOMS, device=next_dist.device)

    # Distribute probability mass from next distribution to projected distribution
    proj_distribution = projected_dist.view(-1)
    proj_distribution.index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
    proj_distribution.index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
    
    return projected_dist

class CategoricalQNetwork(nn.Module):
    def __init__(self, input_shape, num_actions, num_atoms=51):
        super(CategoricalQNetwork, self).__init__()
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
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions * num_atoms)  # Output for each atom-action pair
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size(0), -1)
        logits = self.fc(conv_out).view(-1, self.num_actions, self.num_atoms)
        probs = torch.softmax(logits, dim=2)  # Apply softmax across atoms for each action
        return probs

class CategoricalDQNAgent:
    def __init__(self, input_shape, num_actions, writer, warmup_steps=50_000, device='cuda'):
        self.q_network = CategoricalQNetwork(input_shape, num_actions)
        self.q_network = self.q_network.to(device)
        self.q_network.train()

        self.target_network = CategoricalQNetwork(input_shape, num_actions)
        self.target_network = self.target_network.to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.train()
        
        self.num_actions = num_actions
        self.epsilon_schedule = get_epsilon_schedule(start=1.0, end=0.1)
        self.memory = ReplayMemory(200_000)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-5)
        self.loss_fn = nn.MSELoss()
        self.gamma = 0.95
        self._warmup_steps = warmup_steps
        self.device = device
        self.writer = writer
    
    def act(self, state, step_cnt):
        if step_cnt < self._warmup_steps:
            action = torch.randint(self.num_actions, (1,)).item()
        else:
            epsilon = next(self.epsilon_schedule)
            self.writer.add_scalar('Train/Epsilon', epsilon, step_cnt)

            if random.random() < epsilon:
                action = random.choice(range(self.num_actions))
            else:
                with torch.no_grad():
                    if len(state.shape) == 3:
                        state = state.unsqueeze(0)
                    dist = self.q_network(state)
                    dist = dist * atoms
                    action = dist.sum(2).argmax(1).item()
        
        return action
    
    def act_greedy(self, state):
        with torch.no_grad():
            if len(state.shape) == 3:
                state = state.unsqueeze(0)
            dist = self.q_network(state)
            dist = dist * atoms
            action = dist.sum(2).argmax(1).item()
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state.detach(), action, reward, next_state.detach(), done)
    
    def update(self, batch_size, step_cnt):
        if len(self.memory) < batch_size:   
            return
        
        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(self.device)     
        next_states = torch.stack(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        dist = self.q_network(states)
        dist = dist[range(batch_size), actions]

        with torch.no_grad():
            next_dist = self.target_network(next_states)
            next_actions = next_dist.sum(2).argmax(1)
            next_dist = next_dist[range(batch_size), next_actions]
            target_dist = project_distribution(next_dist, rewards, dones, self.gamma)
    
        loss = -torch.sum(target_dist * torch.log(dist + 1e-8), dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if step_cnt % 5000 == 0:
            self.writer.add_scalar('Train/Loss', loss.item(), step_cnt)
            self.writer.flush()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())