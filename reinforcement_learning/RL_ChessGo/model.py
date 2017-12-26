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
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

    def choose_action(self, observation):
        observation = Variable(torch.Tensor(observation[0].astype(np.float32)).permute(2,1,0).unsqueeze(0))
        actions_prob = self.net(observation)
        actions_prob = torch.squeeze(actions_prob).data.numpy()
        #action = actions_prob.squeeze().max(0)[1].data.numpy()[0] 
        action = np.random.choice(range(actions_prob.shape[0]), p=actions_prob) 
        #from IPython import embed;embed()
        #action = actions_prob.data.max()
        return action+1

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a-1)
        self.ep_rs.append(r)

    def learn(self):
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        #from IPython import embed;embed()
        observation_list = Variable(torch.Tensor(np.vstack(self.ep_obs).astype(np.float32)).permute(0,3,1,2))
        #observation_list = observation_list.permute(0,1,2,3)
        actions = Variable(torch.LongTensor(np.array(self.ep_as)))
        
        actions_prob = self.net(observation_list)
        
        #from IPython import embed;embed()
        actions_prob_selected = actions_prob.gather(1, actions.unsqueeze(1))
        
        part1 = actions_prob_selected.squeeze()
        part2 = Variable(torch.Tensor(discounted_ep_rs_norm))
        loss = -torch.mean(part1 * part2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs