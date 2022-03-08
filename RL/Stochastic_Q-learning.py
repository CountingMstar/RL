import gym
import numpy as np
import matplotlib.pyplot as plt
import random as pr

# 게임환경 커스터마이즈
gym.envs.register(
     id='FrozenLake-v0',
     entry_point='gym.envs.toy_text:FrozenLakeEnv',
     kwargs={'map_name': '4x4',
            'is_slippery': True}
)

env = gym.make('FrozenLake-v0')

# Q 지정
Q = np.zeros([env.observation_space.n, env.action_space.n])

num_episodes = 2000
dis = 0.99
learning_rate = 0.85

rList = []
for i in range(num_episodes):
    state = env.reset() # state를 '0'으로 리셋
    rAll = 0
    done = False

#    e = 1. / ((i//100)+1) 
    while not done:
        # E-greedy
 #       if np.random.rand(1) < e:
 #           action = env.action_space.sample()
 #       else:
 #           action = np.argmax(Q[state,:])

        # random noise
        action = np.argmax(Q[state,:] + np.random.randn(1, env.action_space.n) / (i + 1))

        new_state, reward, done,_ = env.step(action) # env.step(action) : 엑션을 취해줌

        Q[state, action] = (1-learning_rate)*Q[state, action] + learning_rate*(reward + dis*np.max(Q[new_state,:]))
        # Q(s,a) = (1-alpha)*Q(s,a) + alpha*(r + gamma*maxQ(s',a'))

        rAll += reward
        state = new_state

    rList.append(rAll)

print('Success rate' + str(sum(rList)/num_episodes))
print('Final Q-Table Values')
print(Q)
