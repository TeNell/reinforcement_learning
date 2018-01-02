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


MEMORY_CAPACITY = 500
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
    
class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, 4 * 2 + 2))     
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

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % 100 == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, 128)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :4]))
        b_a = Variable(torch.LongTensor(b_memory[:, 4:4+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, 4+1:4+2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -4:]))
        
        #from IPython import embed;embed()
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + 0.9 * q_next.max(1)[0].view(128, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn = DQN()

for i_episode in range(10000):
    observation = env.reset()
    ep_r = 0
    while True:
        env.render()
        action = dqn.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        dqn.store_transition(observation, action, reward, observation_)

        ep_r += reward
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done: print('Ep: ', i_episode, '| Ep_r: ', round(ep_r, 2))

        if done:
            break
        observation = observation_