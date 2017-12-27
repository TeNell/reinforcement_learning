"""

Model of policy gradient

Author. Nell (dongboxiang.nell@gmail.com)
Repos. https://github.com/TeNell 

"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    def __init__(self,):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, 3, stride=1, padding=2)
        self.pool1 = nn.AvgPool2d(3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(529, 64)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(64, 8)
        self.sm = nn.Softmax()
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu(x)
        x = self.fc1(x.view(x.size()[0],-1))
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.sm(x)
        return x

class PolicyGradient:
    def __init__(self,):
        self.net = Net()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-2)
        self.gamma = 0.95
        self.vessel_observation, self.vessel_action, self.vessel_reward = [], [], []

    def choose_action(self, observation):
        observation = Variable(torch.Tensor(observation[0].astype(np.float32)).permute(2,1,0).unsqueeze(0))
        actions_prob = self.net(observation)
        actions_prob = torch.squeeze(actions_prob).data.numpy()
        action = np.random.choice(range(actions_prob.shape[0]), p=actions_prob) 
        return action+1

    def store_transition(self, observation, action, reward):
        self.vessel_observation.append(observation)
        self.vessel_action.append(action-1)
        self.vessel_reward.append(reward)

    def learn(self):
        reward_normed = self._discount_and_norm_rewards()
        observation_list = Variable(torch.Tensor(np.vstack(self.vessel_observation).astype(np.float32)).permute(0,3,1,2))
        actions = Variable(torch.LongTensor(np.array(self.vessel_action)))
        
        actions_prob = self.net(observation_list)
        actions_prob_selected = actions_prob.gather(1, actions.unsqueeze(1))
        
        part1 = actions_prob_selected.squeeze()
        part2 = Variable(torch.Tensor(reward_normed))
        loss = -torch.mean(part1 * part2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.vessel_observation, self.vessel_action, self.vessel_reward = [], [], []
        

    def _discount_and_norm_rewards(self):
        reward_normed = np.zeros_like(self.vessel_reward)
        sto = 0
        for i in reversed(range(0, len(self.vessel_reward))):
            sto = sto * self.gamma + self.vessel_reward[i]
            reward_normed[i] = sto

        reward_normed -= np.mean(reward_normed)
        reward_normed /= np.std(reward_normed)
        return reward_normed