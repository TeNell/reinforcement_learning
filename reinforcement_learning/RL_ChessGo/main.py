"""

Solve simple-Go problem using Policy Gradient.

Author. Nell (dongboxiang.nell@gmail.com)
Repos. https://github.com/TeNell 

"""
import random
from env_Go_v2 import Chess_Go
from model import PolicyGradient

MAXITER = 200

RENDER = False
env = Chess_Go(board_size = 21)

PG = PolicyGradient()

for i_episode in range(10000):
    observation = env.reset()
    while True:
        if RENDER: env.render()
        action = PG.choose_action(observation)
        observation_, reward, done = env.step(action)
        PG.store_transition(observation, action, reward)
        
        if done:
            reward_end = reward
            if iter_i > 200: RENDER = True
            print("iter:", iter_i, "  reward:", reward_end)            
            PG.learn()
            break

        observation = observation_


