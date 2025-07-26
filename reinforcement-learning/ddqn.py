from dqn import DQNAgent
import torch


class DDQNAgent(DQNAgent):
    def update(self, batch_size, step_cnt):
        if len(self.memory) < batch_size:   
            return  # Skip update if memory is insufficient

        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(self.device)     
        next_states = torch.stack(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)

        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if step_cnt % 5000 == 0:
            avg_q_value = q_values.mean().detach().cpu().item()  # Free GPU memory by detaching and moving to CPU
            self.writer.add_scalar('Train/Average Q-Value', avg_q_value, step_cnt)
            self.writer.add_scalar('Train/Loss', loss.item(), step_cnt)
            self.writer.flush()
