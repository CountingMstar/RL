import random
from environment import GraphicDisplay, Env

env = Env()

policy_table = [[[0.25, 0.25, 0.25, 0.25]] * env.width
                                    for _ in range(env.height)]
next_policy = policy_table

next_value_table = [[0.00] * env.width
                            for _ in range(env.height)]
value_table = next_value_table

value = -99999
max_index = []
# 반환할 정책 초기화
result = [0.0, 0.0, 0.0, 0.0]
state = [1, 2]
discount_factor = 0.99

def get_value(state):
    # 소숫점 둘째 자리까지만 계산
    return round(value_table[state[0]][state[1]], 2)

# 모든 행동에 대해서 [보상 + (감가율 * 다음 상태 가치함수)] 계산
for index, action in enumerate(env.possible_actions):  # enumerate : 인덱스와 원소로 이루어진 튜플생성
    next_state = env.state_after_action(state, action)
    reward = env.get_reward(state, action)
    next_value = get_value(next_state)
    temp = reward + discount_factor * next_value
    # 받을 보상이 최대인 행동의 index(최대가 복수라면 모두)를 추출
    if temp == value:
        max_index.append(index)
    elif temp > value:
        value = temp
        max_index.clear()
        max_index.append(index)
    
    print(index, action, max_index)
# 행동의 확률 계산
prob = 1 / len(max_index)
for index in max_index:
    result[index] = prob

next_policy[state[0]][state[1]] = result

print(next_policy)