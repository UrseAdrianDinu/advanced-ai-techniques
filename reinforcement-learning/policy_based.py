import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(PolicyNetwork, self).__init__()
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
        conv_out = self.conv(x)
        conv_out = conv_out.view(conv_out.size()[0], -1)
        logits = self.fc(conv_out)
        return F.softmax(logits, dim=1)

class ReinforceAgent:
    def __init__(self, input_shape, num_actions, writer, gamma=0.99, lr=1e-5, device='cuda'):
        self.policy_network = PolicyNetwork(input_shape, num_actions)
        self.policy_network.to(device)
        self.policy_network.train()
        self.gamma = gamma
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.log_probs = []
        self.rewards = []
        self.writer = writer
        self.fp32_err = 2e-07
    
    def act(self, state, step_cnt):
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        probs = self.policy_network(state)
        m = Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        return action.item()
    
    def act_greedy(self, state):
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        probs = self.policy_network(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item()
    
    def store_rewards(self, reward):
        self.rewards.append(reward)

    def compute_returns(self):
        returns = []
        R = 0
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)

        # Here we normalize the returns. This is a rather theoretically unsound
        # trick but it helps with the speed of convergence in this environment.
        returns = (returns - returns.mean()) / (returns.std() + self.fp32_err)

        return returns

    def update_policy(self, step_cnt):
        returns = self.compute_returns()
        log_probs = torch.cat(self.log_probs)
        policy_loss = (-log_probs * returns.to(log_probs.device)).sum()

        self.writer.add_scalar('Policy Loss', policy_loss.item(), step_cnt)


        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        del self.rewards[:]
        del self.log_probs[:]

class ActorCriticPolicy(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(ActorCriticPolicy, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
            )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc_policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        self.fc_value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x)
        conv_out = conv_out.view(conv_out.size()[0], -1)
        logits = self.fc_policy(conv_out)
        value = self.fc_value(conv_out)
        return F.softmax(logits, dim=1), value

class A2CAgent:
    def __init__(self, input_shape, num_actions, writer, gamma=0.99, lr=1e-5, beta=0.01, device='cuda'):
        self.policy_network = ActorCriticPolicy(input_shape, num_actions)
        self.policy_network.to(device)
        self.policy_network.train()
        self.gamma = gamma
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.log_probs = []
        self.values = []
        self.entropies = []
        self.rewards = []
        self.writer = writer
        self.fp32_err = 2e-07
        self.device = device
        self.beta = beta
    
    def act(self, state, step_cnt):
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        probs, value = self.policy_network(state)
        m = Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        self.values.append(value.squeeze(1))
        self.entropies.append(m.entropy())
        return action.item()
    
    def act_greedy(self, state):
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        probs, value = self.policy_network(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item()
    
    def store_rewards(self, reward):
        self.rewards.append(reward)
    
    def compute_returns(self, done, next_state):
        returns = []
        with torch.no_grad():
            if done:
                R = 0
            else:
                _, next_value = self.policy_network(next_state.unsqueeze(0))
                R = next_value.squeeze(1).item()
        
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + self._fp32_err)
        return returns  # ఠ_ఠ
    
    def update_policy(self, step_cnt, done, next_state):
        returns = self.compute_returns(done, next_state)

        log_probs = torch.cat(self.log_probs)
        values = torch.cat(self.values)
        entropies = torch.cat(self.entropies)

        advantage = returns - values

        policy_loss = (-log_probs * advantage.detach()).sum()
        critic_loss = F.smooth_l1_loss(values, returns)
        entropy_loss = -self.beta * entropies.mean()
        loss = policy_loss + critic_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Logging
        self.writer.add_scalar('Overall Loss', loss.item(), step_cnt)

        # Clear memory
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.entropies.clear()

class A2CGAEAgent(A2CAgent):
    def compute_returns(self, done, next_state, tau=0.95):
        with torch.no_grad():
            _, next_value = self.policy_network(next_state.unsqueeze(0))
            values = self.values + [next_value]
        
        returns = []
        gae = 0

        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t+1] * (1 - done) - values[t]
            gae = delta + self.gamma * tau * (1 - done) * gae
            returns.insert(0, gae + values[t])
        
        return torch.tensor(returns).to(self.device)





