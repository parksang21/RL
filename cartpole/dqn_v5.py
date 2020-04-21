import sys
import numpy as np
import tensorflow.compat.v1 as tf
import random
import gym
from collections import deque

class DQN:
    def __init__(self, env, multistep=False):

        tf.disable_eager_execution()

        self.env = env

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.multistep = multistep  # Multistep(n-step) 구현 시 True로 설정, 미구현 시 False
        self.n_steps = 4            # Multistep(n-step) 구현 시 n 값, 수정 가능

        self.replay = 32
        self.replay_Memory = []
        self.minibatch = 64
        self.learning_rate = 0.001
        self.discount = 0.9
        self.epsilon = 0.1


    def train_minibatch(self, ):
        # mini batch를 받아 policy를 update
        pass

    def update_epsilon(self, epsilon):
        # Exploration 시 사용할 epsilon 값을 업데이트
        return epsilon * 0.9999

    def remember_state(self, state, action, reward, next_state, done, count):
        self.replay_Memory.append([state, action, reward, next_state, done, count])
        if len(self.replay_Memory) > 2000:
            del self.replay_Memory[0]

    # episode 최대 횟수는 구현하는 동안 더 적게, 더 많이 돌려보아도 무방합니다.
    # 그러나 평가시에는 episode 최대 회수를 1500 으로 설정합니다.
    def learn(self, max_episode=1500):
        avg_step_count_list = []     # 결과 그래프 그리기 위해 script.py 로 반환
        last_100_episode_step_count = deque(maxlen=100)
        x = tf.placeholder(dtype=tf.float32, shape=(None, self.state_size))

        y = tf.placeholder(dtype=tf.float32, shape=(None, self.action_size))
        dropout = tf.placeholder(dtype=tf.float32)

        # Main 네트워크
        with tf.variable_scope("Network_M{}".format(int(self.multistep))):
            W1 = tf.get_variable('W1', shape=[self.state_size, 200], initializer=tf.initializers.glorot_normal())
            W2 = tf.get_variable('W2', shape=[200, 200], initializer=tf.initializers.glorot_normal())
            W3 = tf.get_variable('W3', shape=[200, self.action_size], initializer=tf.initializers.glorot_normal())

            b1 = tf.Variable(tf.zeros([1], dtype=tf.float32))
            b2 = tf.Variable(tf.zeros([1], dtype=tf.float32))

            _L1 = tf.nn.relu(tf.matmul(x, W1) + b1)
            L1 = tf.nn.dropout(_L1, dropout)
            _L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
            L2 = tf.nn.dropout(_L2, dropout)
            Q_pre = tf.matmul(L2, W3)

            # Target 네트워크
            W1_r = tf.get_variable('W1_r', shape=[self.state_size, 200])
            W2_r = tf.get_variable('W2_r', shape=[200, 200])
            W3_r = tf.get_variable('W3_r', shape=[200, self.action_size])

            b1_r = tf.Variable(tf.zeros([1], dtype=tf.float32))
            b2_r = tf.Variable(tf.zeros([1], dtype=tf.float32))

            L1_r = tf.nn.relu(tf.matmul(x, W1_r) + b1_r)
            L2_r = tf.nn.relu(tf.matmul(L1_r, W2_r) + b2_r)
            Q_pre_r = tf.matmul(L2_r, W3_r)

            # Loss function 정의
            cost = tf.reduce_sum(tf.square(y - Q_pre))
            optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=1e-8)
            train = optimizer.minimize(cost)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Target 네트워크에 main 네트워크 값을 카피해줌
            sess.run(W1_r.assign(W1))
            sess.run(W2_r.assign(W2))
            sess.run(W3_r.assign(W3))
            sess.run(b1_r.assign(b1))
            sess.run(b2_r.assign(b2))


            global_step = 0
            for episode in range(max_episode):
                done = False
                state = self.env.reset()
                step_count = 0

                self.epsilon = self.update_epsilon(self.epsilon)

                # episode 시작
                while not done:

                    state = np.reshape(state, [1, self.state_size])

                    # greedy 하게 진행
                    if self.epsilon > np.random.rand(1):
                        action = self.env.action_space.sample()
                    else:
                        Q = sess.run(Q_pre, feed_dict={x: state, dropout: 1})
                        action = np.argmax(Q)

                    next_state, reward, done, _ = self.env.step(action)

                    self.remember_state(state, action, reward, next_state, done, step_count)
                    state = next_state
                    step_count += 1
                    global_step += 1

                if global_step < 2000:
                    continue

                if len(self.replay_Memory) > self.replay:
                    indexes = np.random.randint(low=0, high=len(self.replay_Memory) - self.n_steps +1, size=self.minibatch)
                    for index in indexes:
                        state, action, reward, next_state, done, count = self.replay_Memory[index]

                        Q = sess.run(Q_pre, feed_dict={x: state, dropout: 1})

                        discount = self.discount
                        reward_sum = reward
                        if done and count < 200:
                            Q[0, action] = -500
                        else:
                            last_state = next_state

                            if self.multistep:
                                for step in range(1, self.n_steps +1):
                                    n_state, n_action, n_reward, n_next_state, n_done, n_count = self.replay_Memory[index + step]
                                    if n_done and n_count < 200:
                                        reward_sum = -500
                                        break
                                    last_state = n_state
                                    reward_sum += discount * n_reward
                                    discount *= discount

                            n_next_state = np.reshape(last_state, [1, self.state_size])
                            Qn = sess.run(Q_pre_r, feed_dict={x: n_next_state})
                            Q[0, action] = reward_sum + discount * np.max(Qn)

                        sess.run([train, cost], feed_dict={x: state, y: Q, dropout: 1})

                if episode % 5 == 0:
                    sess.run(W1_r.assign(W1))
                    sess.run(W2_r.assign(W2))
                    sess.run(W3_r.assign(W3))
                    sess.run(b1_r.assign(b1))
                    sess.run(b2_r.assign(b2))

                last_100_episode_step_count.append(step_count)


                # 최근 100개의 에피소드 평균 step 횟수를 저장 (이 부분은 수정하지 마세요)
                avg_step_count = np.mean(last_100_episode_step_count)
                print("[Episode {:>5}]  episode step_count: {:>5} avg step_count: {}".format(
                                                                            episode, step_count, avg_step_count))

                avg_step_count_list.append(avg_step_count)

                if avg_step_count > 475:
                    break

        sess.close()
        return avg_step_count_list




import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import gym

orgDQN, multistep = False, False

if len(sys.argv) == 1:
    print("There is no argument, please input")

for i in range(1,len(sys.argv)):
    if sys.argv[i] == "orgDQN":
        orgDQN = True
    elif sys.argv[i] == "multistep":
        multistep = True


env = gym.make('CartPole-v1')

if orgDQN:
    env.reset()
    dqn = DQN(env, multistep=False)
    orgDQN_record = dqn.learn(1500)
    del dqn

if multistep:
    env.reset()
    dqn = DQN(env, multistep=True)
    multistep_record = dqn.learn(1500)
    del dqn

print("Reinforcement Learning Finish")
print("Draw graph ... ")

if orgDQN:
    plt.plot(np.arange((len(orgDQN_record))), orgDQN_record, label='Orginal DQN')
if multistep:
    plt.plot(np.arange((len(multistep_record))), multistep_record, label='Multistep DQN')

plt.legend()
fig =plt.gcf()
plt.savefig("result.png")
plt.show()
