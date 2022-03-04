import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()


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

# These lines establish the feed-forward part of the network used to choose actions
X = tf.placeholder(shape=[1, input_size], dtype=tf.float32) # state input
print(X)
#W = tf.Variable(tf.random_uniform([input_size, output_size], 0, 0.01)) # weight

#Qpred = tf.matmul(X, W) # Out Q prediction
#Y = tf.placeholder(shape=[1, output_size], dtype=tf.float32) # Y label
#
#loss = tf.reduce_sum(tf.square(Y - Qpred))
## cost(W) = (Ws - y)^2
#
#train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
#
#dis = 0.99
#num_episodes = 2000
#
#rList = []
#
#init = tf.global_variables_initializer()
#
#with tf.Session() as sess:
#    sess.run(init)
#    for i in range(num_episodes):
#        s = env.reset()
#        e = 1. / ((i/50)+10)
#        rAll = 0
#        done = False
#        local_loss = []
#
#        while not done:
#            Qs = sess.run(Qpred, feed_dict={X: one_hot(s)})
#
#            # e-greedy
#            if np.random.rand(1) < e:
#                a = env.action_space.sample()
#            else:
#                a = np.argmax(Qs)
#
#            s1, reward, done,_ = env.step(a)
#            
#            # 게임이 끝날때, update Q, and no Qs+1, since it's a terminal state
#            if done:
#                Qs[0, a] = reward
#                # y = r
#
#            # 게임 중간에, obtain the Qs1 values by feeding the new state through our network
#            else:
#                Qs1 = sess.run(Qpred, feed_dict={X: one_hot(s1)})
#                Qs[0, a] = reward + dis*np.max(Qs1)
#                # y = r + gamma*maxQ(s')
#
#            sess.run(train, feed_dict={X: one_hot(s), Y: Qs})
#
#            rAll += reward
#            s = s1
#        rList.append(rAll)
#
#print('Percent of successful episodes:' + str(sum(rList)/num_episodes) + '%')
#plt.bar(range(len(rList)), rList, color='blue')
#plt.show()