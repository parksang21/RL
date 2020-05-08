
import gym
from ma_gym.wrappers import Monitor

env = gym.make("PongDuel-v0")
print(env.reset())

out = env.step(env.action_space.sample())


print('obs_n : {}, reward_n {}, done_n {}, info {}'.format(out[0], out[1], out[2], out[3]))

print(type(out[0]))
