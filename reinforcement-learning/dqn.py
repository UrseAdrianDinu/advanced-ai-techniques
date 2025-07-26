import torch.nn as nn
import numpy as np
import torch
from collections import deque, namedtuple
import random
import itertools
import torch.optim as optim

    

class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    
# Epsilon Schedule
def get_epsilon_schedule(start=1.0, end=0.1, steps=800_000):
    eps_step = (start - end) / steps
    def frange(start, end, step):
        x = start
        while x > end:
            yield x
            x -= step
    return itertools.chain(frange(start, end, eps_step), itertools.repeat(end))

class QNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(QNetwork, self).__init__()
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
            nn.Linear(512, num_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        # Pass through convolutional layers
        conv_out = self.conv(x)
        # print("Conv output shape:", conv_out.size())

        # Flatten the conv output to pass into the fully connected layers
        conv_out = conv_out.view(conv_out.size()[0], -1)
        
        # Pass through fully connected layers
        return self.fc(conv_out)

class DQNAgent:
    def __init__(self, input_shape, num_actions, writer, warmup_steps=50_000, device='cuda'):
        self.q_network = QNetwork(input_shape, num_actions)
        self.q_network = self.q_network.to(device)
        self.q_network.train()
        self.target_network = QNetwork(input_shape, num_actions)
        self.target_network = self.target_network.to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.train()
        self.num_actions = num_actions
        self.epsilon_schedule = get_epsilon_schedule(start=1.0, end=0.1)
        self.memory = ReplayMemory(200_000)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-5)
        self.loss_fn = nn.MSELoss()
        self.gamma = 0.95
        self._warmup_steps = warmup_steps  # Warm-up steps
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
                    if len(state.shape) == 3:  # If state is (4, 84, 84)
                        state = state.unsqueeze(0)  # Add batch dimension
                    q_values = self.q_network(state)
                    action = q_values.argmax().item()
        
        return action
    
    def act_greedy(self, state):
        """Select the action with the highest Q-value without any exploration."""
        with torch.no_grad():
            if len(state.shape) == 3:  # Ensure the input is batch-ready
                state = state.unsqueeze(0)  # Add batch dimension if needed
            q_values = self.q_network(state)
            action = q_values.argmax().item()  # Select action with highest Q-value
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state.detach(), action, reward, next_state.detach(), done)

    def update(self, batch_size, step_cnt):
        if len(self.memory) < batch_size:   
            return  # Skip update if memory is insufficient

        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert states and next_states to PyTorch tensors
        states = torch.stack(states).to(self.device)        # Shape: (batch_size, 4, 84, 84)
        next_states = torch.stack(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute Q-values for the current states and selected actions
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)  # Shape: (batch_size)
        
        # Compute Q-values for the next states
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
        
        # Calculate target Q-values
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss and backpropagate
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if step_cnt % 5000 == 0:
            avg_q_value = q_values.mean().detach().cpu().item()  # Free GPU memory by detaching and moving to CPU
            self.writer.add_scalar('Train/Average Q-Value', avg_q_value, step_cnt)
            self.writer.add_scalar('Train/Loss', loss.item(), step_cnt)
            self.writer.flush()
        
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
