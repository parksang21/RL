'''
2020710425 박상욱, Sang Uk Park
2020.04.05 Deep_Reinforcement_Learning_Assignment 1,
'''

# import numpy
import numpy as np


class Agent():

    def __init__(self, Q, env, mode):
        """
        initialize values
        :param Q: Q table
        :param env: open ai gym environment
        :param mode: learning mode or test mode
        """
        self.Q = Q
        self.env = env
        self.mode = mode
        self.epsilon = 0.0

    def select_action(self, state):
        """
        choose action w.r.t state and modes
        :param state: current states
        :return: action
        """
        if self.mode == "learning_mode":
            # if the mode is 'learning_mode' use epsilon greedy to optimize
            if self.epsilon <= np.random.uniform(0, 1):
                # greedy part
                action = np.argmax(self.Q[state])
            else:
                # epsilon part
                action = np.random.choice(4)
        else:
            # test mode
            action = np.argmax(self.Q[state])

        return action

    def learn(self):
        """
        Q-Learning part
        """

        # initialize hyper parameters
        max_episodes = 10000
        alpha = 0.1
        gamma = 0.8
        rewards = []

        for episode in range(1, max_episodes + 1):

            state = self.env.reset()
            reward_e = 0
            self.epsilon = 0.3 * (1 / episode)
            is_done = False

            while not is_done:
                action = self.select_action(state)
                next_state, reward, is_done, info = self.env.step(action)

                if reward == 0:
                    if is_done:
                        learning_reward = -1
                    else:
                        learning_reward = -0.1
                else:
                    learning_reward = reward

                reward_e += reward
                q = self.Q[state][action]
                self.Q[state][action] += alpha * (learning_reward + gamma * (np.amax(self.Q[next_state])) - q)

                if is_done:
                    rewards.append(reward_e)
                    break

                state = next_state

            if not episode % 100:
                history = rewards[(episode-100):episode]
                avg_reward = sum(history)/100
                print("\rEpisode {}/{} || average reward {}".format(episode, max_episodes, avg_reward), end="")