import numpy as np


class FinalModel:
    def __init__(self, env):
        # Load your Model here
        self.action_size = env.action_space.shape[0]

    def get_action(self, state):
        # change to your model
        return np.random.rand(self.action_size) * 2 - 1