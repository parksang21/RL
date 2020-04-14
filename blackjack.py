import gym
import numpy as np
import torch

env = gym.make('Blackjack-v0')
def run_episode(env, Q, epsilon, n_action):

    state = env.reset()
    rewards = []
    actions = []
    states = []
    is_done = False

    while not is_done:

        probs = torch.ones(n_action) * epsilon / n_action
        best_action = torch.argmax(Q[state]).item()
        probs[best_action] += 1.0 - epsilon
        action = torch.multinomial(probs, 1).item()
        actions.append(action)
        states.append(state)
        state, reward, is_done, info = env.step(action)
        rewards.append(reward)
        if is_done:
            break
            
    return states, actions, rewards

from collections import defaultdict

def mc_control_epsilon_greedy(env, gamma, n_episode, epsilon):
    n_action = env.action_space.n
    G_sum = defaultdict(float)
    N = defaultdict(int)
    Q = defaultdict(lambda : torch.empty(n_action))

    for episode in range(n_episode):
        states_t, actions_t, rewards_t = run_episode(env, Q, epsilon, n_action)
        return_t = 0
        G = {}

        for state_t, action_t, reward_t in zip(states_t[::-1], actions_t[::-1], rewards_t[::-1]):
            return_t = gamma * return_t + reward_t
            G[(state_t, action_t)] = return_t

        for state_action, return_t in G.items():
            state, action = state_action
            if state[0] <= 21:
                G_sum[state_action] += return_t
                N[state_action] += 1
                Q[state][action] = G_sum[state_action] / N[state_action]

        policy = {}

        for state, actions in Q.items():
            policy[state] = torch.argmax(actions).item()

        print("episode #", episode, " is done")

    return Q, policy


def simulate_episode(env, policy):
    state = env.reset()
    is_done = False
    while not is_done:
        action = policy[state]
        state, reward, is_done, info = env.step(action)
        if is_done:
            return reward


gamma = 1
n_episode = 50000
epsilon = 0.1

optimal_Q, optimal_policy = mc_control_epsilon_greedy(env, gamma, n_episode, epsilon)

n_episode = 10000
n_win_optimal = 0
n_loose_optimal = 0

for _ in range(n_episode):
    reward = simulate_episode(env, optimal_policy)
    if reward == 1:
        n_win_optimal += 1
    elif reward == -1:
        n_loose_optimal += 1

print("Win\t{}\nLoose\t{}".format(n_win_optimal, n_loose_optimal))