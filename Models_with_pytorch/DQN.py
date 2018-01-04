"""

Author. Nell (dongboxiang.nell@gmail.com)
Repos. https://github.com/TeNell 

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import gym
import numpy as np

env = gym.make('CartPole-v0')
env = env.unwrapped


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class DQN:
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.exchange_netparams_iter = 0
        self.storage_limit = 200
        self.batch_size = 128
        self.storage_full = False
               
        self.storage = {'observation':[],'action':[],'reward':[],'observation_':[]}
        
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.01)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        if np.random.uniform() < 0.9:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]
        else:
            action = np.random.randint(0, 2)
        return action

    def store_states(self, observation, action, reward, observation_):
       
        self.storage['observation'].append(observation)
        self.storage['action'].append(action)
        self.storage['reward'].append(reward)
        self.storage['observation_'].append(observation_)
        
        if len(self.storage['observation']) > self.storage_limit:
            self.storage_full = True
            self.storage['observation'].pop(0)
            self.storage['action'].pop(0)
            self.storage['reward'].pop(0)
            self.storage['observation_'].pop(0)

    def learn(self):
        if self.exchange_netparams_iter % 100 == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.exchange_netparams_iter += 1

        idx = np.random.choice(self.storage_limit, self.batch_size)
        batch_observation = Variable(torch.FloatTensor(np.array(self.storage['observation'])[idx]))
        batch_action = Variable(torch.LongTensor(np.array(self.storage['action'])[idx]))
        batch_reward = Variable(torch.FloatTensor(np.array(self.storage['reward'])[idx]))
        batch_observation_ = Variable(torch.FloatTensor(np.array(self.storage['observation_'])[idx]))
        
        q_eval = self.eval_net(batch_observation).gather(1, batch_action.unsqueeze(1))
        
        q_next = self.target_net(batch_observation_).detach()
        q_target = (batch_reward + 0.9 * q_next.max(1)[0]).unsqueeze(1)
        
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn = DQN()

step = 1
while True:
    observation = env.reset()
    rewards = 0
    while True:
        env.render()
        action = dqn.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        
        dqn.store_states(observation, action, reward, observation_)
        rewards += reward
        
        if dqn.storage_full:
            dqn.learn()
            if done: print('step:{0:<5d}rewards:{1:<1.3f}'.format(step, rewards)); step += 1

        if done: break
        observation = observation_
        

        

