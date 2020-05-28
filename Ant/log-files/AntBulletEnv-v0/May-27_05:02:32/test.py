import gym
import pybullet_envs


if __name__ == '__main__':

    env = gym.make("AntBulletEnv-v0")

    env.render()
    env.reset()
    while True:
        env.render()
        action = env.action_space.sample()
        next_s, reward, done, _ = env.step(action)