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

        self.replay = 10
        self.replay_Memory = []
        self.minibatch = 50
        self.learning_rate = 0.01
        self.discount = 0.99
        self.epsilon = 0

        # Train Network 구성
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, self.state_size))
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, self.action_size))
        self.dropout = tf.placeholder(dtype=tf.float32)

        self.W1 = tf.get_variable('W1', shape=[self.state_size, 200], initializer=tf.initializers.glorot_normal())
        self.W2 = tf.get_variable('W2', shape=[200, 200], initializer=tf.initializers.glorot_normal())
        self.W3 = tf.get_variable('W3', shape=[200, self.action_size], initializer=tf.initializers.glorot_normal())

        self.b1 = tf.Variable(tf.zeros([1], dtype=tf.float32))
        self.b2 = tf.Variable(tf.zeros([1], dtype=tf.float32))

        self._L1 = tf.nn.relu(tf.matmul(self.x, self.W1) + self.b1)
        self.L1 = tf.nn.dropout(self._L1, self.dropout)
        self._L2 = tf.nn.relu(tf.matmul(self.L1, self.W2) + self.b2)
        self.L2 = tf.nn.dropout(self._L2, self.dropout)
        self.Q_pre = tf.matmul(self.L2, self.W3)


        # Target Network 구성
        self.W1_r = tf.get_variable('W1_r', shape=[self.state_size, 200])
        self.W2_r = tf.get_variable('W2_r', shape=[200, 200])
        self.W3_r = tf.get_variable('W3_r', shape=[200, self.action_size])

        self.b1_r = tf.Variable(tf.zeros([1], dtype=tf.float32))
        self.b2_r = tf.Variable(tf.zeros([1], dtype=tf.float32))

        self.L1_r = tf.nn.relu(tf.matmul(self.x, self.W1_r) + self.b1_r)
        self.L2_r = tf.nn.relu(tf.matmul(self.L1_r, self.W2_r) + self.b2_r)
        self.Q_pre_r = tf.matmul(self.L2_r, self.W3_r)

        self.cost = tf.reduce_sum(tf.square(self.y - self.Q_pre))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=0.01)
        self.train = self.optimizer.minimize(self.cost)

    def _build_network(self, ):
        # Target 네트워크와 Local 네트워크를 설정
        pass

    def predict(self, state):
        # state를 넣어 policy에 따라 action을 반환
        return self.model(np.atleast_2d(state.astype('float')))

    def train_minibatch(self, ):
        # mini batch를 받아 policy를 update
        pass

    def update_epsilon(self, n_episode):
        # Exploration 시 사용할 epsilon 값을 업데이트
        return 1. / ((n_episode / 25) + 1)
        

    def remember_state(self, state, action, reward, next_state, done, count):
        self.replay_Memory.append([state, action, reward, next_state, done, count])

        # replay memory가 5만이 넘으면 FIFO를 통해 제일 앞의 memroy 삭제
        if len(self.replay_Memory) > 50000:
            del self.replay_Memory[0]

    # episode 최대 횟수는 구현하는 동안 더 적게, 더 많이 돌려보아도 무방합니다.
    # 그러나 평가시에는 episode 최대 회수를 1500 으로 설정합니다.
    def learn(self, max_episode=1500):
        avg_step_count_list = []     # 결과 그래프 그리기 위해 script.py 로 반환
        last_100_episode_step_count = deque(maxlen=100)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            sess.run(self.W1_r.assign(self.W1))
            sess.run(self.W2_r.assign(self.W2))
            sess.run(self.W3_r.assign(self.W3))
            sess.run(self.b1_r.assign(self.b1))
            sess.run(self.b2_r.assign(self.b2))


            for episode in range(max_episode):
                done = False
                state = self.env.reset()
                step_count = 0

                e = self.update_epsilon(episode)

                # episode 시작
                while not done:

                    state = np.reshape(state, [1, self.state_size])

                    Q = sess.run(self.Q_pre, feed_dict={self.x:state, self.dropout:1})
                    # greedy 하게 진행
                    if e > np.random.rand(1):
                        action = self.env.action_space.sample()
                    else:
                        action = np.argmax(Q)


                    next_state, reward, done, _ = self.env.step(action)
                    self.remember_state(state, action, reward, next_state, done, step_count)
                    state = next_state
                    step_count += 1

                if not self.multistep:
                    self.n_steps = 1

                if episode % self.n_steps == 0 and len(self.replay_Memory) > self.minibatch:
                    for sample in random.sample(self.replay_Memory, self.replay):
                        state_r, action_r, reward_r, next_state_r, done_r, count_r = sample

                        Q = sess.run(self.Q_pre, feed_dict={self.x: state_r, self.dropout: 1})

                        if done_r:
                            if count_r < 200:
                                Q[0, action_r] = -100
                        else:
                            _next_state_r = np.reshape(next_state_r, [1, self.state_size])
                            Q1 = sess.run(self.Q_pre_r, feed_dict={self.x: _next_state_r})
                            Q[0, action_r] = reward_r + self.discount * np.max(Q1)

                        _, loss = sess.run([self.train, self.cost], feed_dict={self.x: state_r, self.y: Q, self.dropout: 1})

                    sess.run(self.W1_r.assign(self.W1))
                    sess.run(self.W2_r.assign(self.W2))
                    sess.run(self.W3_r.assign(self.W3))
                    sess.run(self.b1_r.assign(self.b1))
                    sess.run(self.b2_r.assign(self.b2))

                last_100_episode_step_count.append(step_count)


                # 최근 100개의 에피소드 평균 step 횟수를 저장 (이 부분은 수정하지 마세요)
                avg_step_count = np.mean(last_100_episode_step_count)
                print("[Episode {:>5}]  episode step_count: {:>5} avg step_count: {}".format(episode, step_count, avg_step_count))

                avg_step_count_list.append(avg_step_count)

        return avg_step_count_list
