import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam


# 게임환경 커스터마이즈 
gym.envs.register(
     id='FrozenLake-v0',
     entry_point='gym.envs.toy_text:FrozenLakeEnv',
     kwargs={'map_name': '4x4',
            'is_slippery': True}
)

env = gym.make('FrozenLake-v0')

# one hot 함수 : 하나만 활성화
def one_hot(x):
    return np.identity(16)[x:x+1]

# input output size based on the env    
input_size = env.observation_space.n
output_size = env.action_space.n
learning_rate = 0.1

def build_model(input_size, output_size):
    model = Sequential()
    model.add(Flatten(input_shape=(1, input_size)))
    model.add(Dense(24, activation = 'relu'))
    model.add(Dense(24, activation = 'relu'))
    model.add(Dense(output_size, activation = 'linear'))
    return model

model = build_model(input_size, output_size)

print(model.summary())