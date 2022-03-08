import gym
import random

env = gym.make('CartPole-v0')
states = env.observation_space.shape[0]
actions = env.action_space.n

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
# Dense: fully connected layer 모듈
# Dense(8, input_dim=4, init='uniform', activation='relu')) = (출력 뉴런수, 입력뉴런수, 가중치 초기화방법, 활성화 함수)
# Flatten: 추출된 주요 특징을 fully connected layer에 전달하기 위해 1차원 자료로 바꿔주는 layer
from tensorflow.keras.optimizers import Adam

def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape = (1, states)))
    model.add(Dense(24, activation = 'relu'))
    # 출력 뉴런수 24개, 활성화 함수 'relu'
    model.add(Dense(24, activation = 'relu'))
    model.add(Dense(actions, activation = 'linear'))
    
    return model

model = build_model(states, actions)
print(model.summary())