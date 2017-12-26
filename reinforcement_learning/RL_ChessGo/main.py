"""



Author. Nell (dongboxiang.nell@gmail.com)
Homepage. https://github.com/TeNell 

"""
import random
from env_Go_v2 import Chess_Go
from model import PolicyGradient
#import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = 2800

RENDER = False  # rendering wastes time
env = Chess_Go(board_size = 21)

#env = env.unwrapped

#print(env.action_space)
#print(env.observation_space)
#print(env.observation_space.high)
#print(env.observation_space.low)

RL = PolicyGradient()
#RL = PolicyGradient(
#    n_actions=env.action_space.n,
#    n_features=env.observation_space.shape[0],
#    learning_rate=0.02,
#    reward_decay=0.995,
    # output_graph=True,
#)
#print(dir(env))

for i_episode in range(4000000):

    observation = env.reset()
    i = 1
    while True:
        #print(i)
        i += 1
        if RENDER: env.render()

        action = RL.choose_action(observation)
        #action = random.choice(range(1,9))
        #print(action)
        observation_, reward, done = env.step(action)     # reward = -1 in all cases

        RL.store_transition(observation, action, reward)

        if done:
            # calculate running reward
            ep_rs_sum = sum(RL.ep_rs)
            #from IPython import embed;embed()
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if i_episode > 200: RENDER = True     # rendering

            print("episode:", i_episode, "  reward:", running_reward)

            vt = RL.learn()  # train

            #if i_episode == 30:
            #    plt.plot(vt)  # plot the episode vt
            #    plt.xlabel('episode steps')
            #    plt.ylabel('normalized state-action value')
            #    plt.show()

            break

        observation = observation_


