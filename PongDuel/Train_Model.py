import tensorflow as tf
import os
import gym
from ma_gym.wrappers import monitor
import time
import numpy as np
import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers as ko
from collections import deque


from gym_to_gif import save_frames_as_gif


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


np.random.seed(1)
tf.random.set_seed(1)

# Neural Network Model Defined at Here.
# Neural Network Model Defined at Here.
class Model(tf.keras.Model):
    def __init__(self, model_num):
        super().__init__(name='basic_ddqn{}'.format(model_num))
        # you can try different kernel initializer
        self.fc1 = kl.Dense(200, activation='relu', kernel_initializer='he_uniform')
        self.fc2 = kl.Dense(200, activation='relu', kernel_initializer='he_uniform')
        self.logits = kl.Dense(3, name='q_values',)

    # forward propagation
    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.logits(x)
        return x

    def action_value(self, obs):
        q_values = self.predict(obs)
        best_action = np.argmax(q_values, axis=-1)
        return best_action if best_action.shape[0] > 1 else best_action[0], q_values[0]

def test_model():
    env = gym.make('PongDuel-v0')
    print('num_actions: ', 3)
    model = Model(3)

    obs = env.reset()
    obs = np.asarray(obs[0] + obs[1][:2])
    print('obs_shape: ', obs.shape)

    # tensorflow 2.0: no feed_dict or tf.Session() needed at all
    best_action, q_values = model.action_value(obs[None])
    print('res of test model: ', best_action, q_values)  # 0 [ 0.00896799 -0.02111824]

class DDQNAgent:  # Double Deep Q-Network
    def __init__(self, modelp, target_modelp, modelq, target_modelq, env, buffer_size=10000, learning_rate=.0015, epsilon=1, epsilon_dacay=0.995,
                 min_epsilon=.01, gamma=.9, batch_size=64, target_update_iter=2, train_nums=10000, start_learning=10,):

        self.model_p = modelp
        self.target_model_p = target_modelp

        self.model_q = modelq
        self.target_model_q = target_modelq

        # gradient clip
        opt = ko.Adam(learning_rate=learning_rate, clipvalue=10.0)
        self.model_p.compile(optimizer=opt, loss='mse')

        opt2 = ko.Adam(learning_rate=learning_rate, clipvalue=10.0)
        self.model_q.compile(optimizer=opt2, loss='mse')

        self.env = env
        self.lr = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_dacay
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_iter = target_update_iter
        self.train_nums = train_nums
        self.num_in_buffer = 0
        self.buffer_size = buffer_size
        self.start_learning = start_learning

        self.obs = np.empty((self.buffer_size,) + (4 * 12,))
        self.actions = np.empty(( self.buffer_size,2), dtype=np.int8)
        self.rewards = np.empty((self.buffer_size, 2), dtype=np.float32)
        self.dones = np.empty((self.buffer_size, 2), dtype=np.bool)
        self.next_states = np.empty((self.buffer_size, ) +(4 * 12,))
        self.next_idx = 0
        self.player, self.opponent = 0, 1

    def train(self):
        # initialize the initial observation of the agent
        average_step_count = deque(maxlen=100)

        for t in range(1, self.train_nums):
            obs = self.reset()
            step_count = 0
            done = [False, False]
            rewards = np.zeros((1,2))
            self.e_decay()
            win_step_count = 0
            obs_window = np.zeros((4 * 12,))
            n_obs_window = np.zeros((4 * 12,))
            while not done[0] and step_count < 100000:
                step_count += 1
                for i in range(12, 3 * 12 + 1, 12):
                    obs_window[i-12:i] = obs_window[i:i+12]
                    n_obs_window[i-12:i] = obs_window[i:i+12]

                obs_window[-12:] = obs

                best_action_p, q_values_p = self.model_p.action_value(obs_window[None])  # input the obs to the network model
                best_action_q, q_values_q = self.model_q.action_value(obs_window[None])  # input the obs to the network model

                action = self.get_action([best_action_p, best_action_q])   # get the real action

                next_obs, reward, done, _ = self.env.step(action)    # take the action in the env to return s', r, done
                # env.render()
                next_obs = np.asarray(next_obs[0] + next_obs[1][:2])
                # print("ff")
                n_obs_window[-12:] = next_obs

                # _____________________________________________________________________________________________________
                # reward function 만들어야 한다.
                win_step_count += 1
                ball_position = obs[2:4]
                ball_d = obs[4:10]
                if reward[0] == 1:
                    # 왼쪽이 이기고 오른쪽이 짐
                    # print("\t++Left")
                    reward[0] = 1
                    reward[1] = -10 * abs(ball_position[0] - obs[-2])
                    win_step_count = 0

                elif reward[1] == 1:
                    # print("\t++Right")
                    reward[1] = 1
                    reward[0] = -10 * abs(ball_position[0] - obs[0])
                    win_step_count = 0

                else:
                    cur_ball_d = obs[4:10]
                    next_ball_d = next_obs[4:10]
                    if abs(ball_position[1] - obs[1]) < 0.1 and abs(ball_position[0] - obs[0]) < 0.1 \
                            and sum(abs(cur_ball_d - next_ball_d)) == 2:
                        # print(abs(ball_position[1] - obs[1]))
                        # print(abs(ball_position[0] - obs[0]))
                        # print("++Left")
                        reward[0] = 1
                    elif abs(ball_position[1] - obs[11]) < 0.1 and abs(ball_position[0] - obs[10]) < 0.1 \
                            and sum(abs(cur_ball_d - next_ball_d)) == 2:
                        # print(abs(ball_position[1] - obs[11]))
                        # print(abs(ball_position[0] - obs[10]))
                        # print("++Right")
                        reward[1] = 1
                    else:
                        reward[0] = -1 * np.sqrt((obs[0] - ball_position[0])*(obs[0] - ball_position[0]))
                        reward[1] = -1 * np.sqrt((obs[10] - ball_position[0])*(obs[10] - ball_position[0]))

                # _____________________________________________________________________________________________________

                self.store_transition(obs_window, action, reward, n_obs_window, done)
                self.num_in_buffer = min(self.num_in_buffer + 1, self.buffer_size)
                rewards += np.asarray(reward)

                if t % self.target_update_iter == 0 and t > self.start_learning:
                    self.update_target_model()

                if all(done):
                    obs = self.reset()
                else:
                    obs = next_obs

            if t > self.start_learning:  # start learning
                losses = self.train_step()
                print("[Total EPISODE{:>5}]\tsteps : {:>5}\tavg100 setp : {:>5.5}\tlosses: {}\tepsilon: {}\nrewards : {}"
                      .format(t, step_count, np.mean(average_step_count), losses, self.epsilon, rewards))

            average_step_count.append(step_count)

            if t % 100 == 0:
                self.model_p.save_weights('./l1/left.ckpt')
                self.model_q.save_weights('./r1/right.ckpt')

    def train_step(self):
        idxes = self.sample(self.batch_size)
        s_batch = self.obs[idxes]
        a_batch = self.actions[idxes]
        r_batch = self.rewards[idxes]
        ns_batch = self.next_states[idxes]

        best_action_idxes_p, _ = self.model_p.action_value(ns_batch)
        best_action_idxes_q, _ = self.model_q.action_value(ns_batch)

        target_q_p = self.get_target_value(ns_batch, 0)
        target_q_q = self.get_target_value(ns_batch, 1)

        target_q_p = r_batch[:,0] + self.gamma * target_q_p[np.arange(target_q_p.shape[0]), best_action_idxes_p]
        target_q_q = r_batch[:,1] + self.gamma * target_q_q[np.arange(target_q_q.shape[0]), best_action_idxes_q]

        target_f_p = self.model_p.predict(s_batch)
        target_f_q = self.model_q.predict(s_batch)
        for i, val in enumerate(a_batch):
            target_f_p[i][val] = target_q_p[i]
            target_f_q[i][val] = target_q_q[i]

        losses_p = self.model_p.train_on_batch(s_batch, target_f_p)
        losses_q = self.model_q.train_on_batch(s_batch, target_f_q)

        return losses_p, losses_q

    def evalation(self, env, render=True):
        obs, done, ep_reward = self.reset(), [False, False], 0
        # one episode until done

        frame=[]
        while not all(done):
            action_p, _ = self.model_p.action_value(obs[None])  # Using [None] to extend its dimension (4,) -> (1, 4)
            action_q, _ = self.model_q.action_value(obs[None])  # Using [None] to extend its dimension (4,) -> (1, 4)
            obs, reward, done, info = self.env.step([action_p, action_q])
            obs = np.asarray(obs[0] + obs[1][:2])
            ep_reward += np.sum(reward)
            if render:  # visually show
                frame.append(env.render(mode='rgb_array'))

        # env.close()
        global b
        save_frames_as_gif(frame, path='./render_results/', filename="{}trial.gif".format(b))
        print(b, 'is saved')
        b += 1
        return ep_reward

    def store_transition(self, obs, action, reward, next_state, done):
        n_idx = self.next_idx % self.buffer_size
        self.obs[n_idx] = obs
        self.actions[n_idx] = action
        self.rewards[n_idx] = reward
        # next_state = np.asarray(next_state[0] + next_state[1][:2])
        self.next_states[n_idx] = np.asarray(next_state)
        self.dones[n_idx] = done
        self.next_idx = (self.next_idx + 1) % self.buffer_size

    def sample(self, n):
        assert n < self.num_in_buffer
        res = []
        while True:
            num = np.random.randint(0, self.num_in_buffer)
            if num not in res:
                res.append(num)
            if len(res) == n:
                break
        return res


    def get_action(self, best_action):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return best_action

    # assign the current network parameters to target network
    def update_target_model(self):
        self.target_model_p.set_weights(self.model_p.get_weights())
        self.target_model_q.set_weights(self.model_q.get_weights())

    def get_target_value(self, obs, p):
        if p == 0:
            return self.target_model_p.predict(obs)
        else:
            return self.target_model_q.predict(obs)

    def e_decay(self):
        if self.min_epsilon < self.epsilon:
            self.epsilon *= self.epsilon_decay

    def reset(self):
        state = self.env.reset()
        state = np.asarray(state[0] + state[1][:2])
        return state

    def step(self, action, p):
        player = p
        next_obs, reward, done, info = self.env.step(action)
        state = np.asarray(next_obs[0] + next_obs[1][:2])

        return state, reward, done, info

if __name__ == "__main__":
    # test_model()
    env = gym.make("PongDuel-v0")
    num_actions = 3
    model = Model(num_actions)
    target_model = Model(num_actions)

    modelq = Model(num_actions)
    target_modelq = Model(num_actions)
    agent = DDQNAgent(model, target_model, modelq, target_modelq, env)

    agent.train()


