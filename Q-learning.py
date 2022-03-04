import gym
import numpy as np
import matplotlib.pyplot as plt
import random as pr

# random argmax 함수
def rargmax(vector):
    m = np.amax(vector) # np.amax : 최댓값 반환
    indices = np.nonzero(vector==m)[0] # np.nonzero : '0'이 아닌값들만 반환
    return pr.choice(indices) # random.choice : 원소를 아무거나 뽑아줌

# 게임환경 커스터마이즈
gym.envs.register(
     id='FrozenLake-v3',
     entry_point='gym.envs.toy_text:FrozenLakeEnv',
     kwargs={'map_name': '4x4',
            'is_slippery': False}
)

# 게임 환경설정
env = gym.make('FrozenLake-v3')

# Q 지정
Q = np.zeros([env.observation_space.n, env.action_space.n])

num_episodes = 2000
dis = 0.99

rList = []
for i in range(num_episodes):
    state = env.reset() # state를 '0'으로 리셋
    rAll = 0
    done = False

#    e = 1. / ((i//100)+1) 
    while not done:
        # E-greedy
#        if np.random.rand(1) < e:
#            action = env.action_space.sample()
#        else:
#            action = np.argmax(Q[state,:])

        # random noise
        action = np.argmax(Q[state,:] + np.random.randn(1, env.action_space.n) / (i + 1))

        new_state, reward, done,_ = env.step(action) # env.step(action) : 엑션을 취해줌

        Q[state, action] = reward + dis*np.max(Q[new_state,:])
        # Q(s,a) = r + gamma*maxQ(s',a')

        rAll += reward
        state = new_state

    rList.append(rAll)

print('Success rate' + str(sum(rList)/num_episodes))
print('Final Q-Table Values')
print(Q)
