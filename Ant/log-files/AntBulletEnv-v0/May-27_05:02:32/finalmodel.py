import numpy as np
from policy import Policy

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
global step
step = 0

class Scaler(object):
    """ Generate scale and offset based on running mean and stddev along axis=0

        offset = running mean
        scale = 1 / (stddev + 0.1) / 3 (i.e. 3x stddev = +/- 1.0)
    """

    def __init__(self, obs_dim):
        """
        Args:
            obs_dim: dimension of axis=1
        """
        self.vars = np.zeros(obs_dim)
        self.means = np.zeros(obs_dim)
        self.m = 0
        self.n = 0
        self.first_pass = True

    def update(self, x):
        """ Update running mean and variance (this is an exact method)
        Args:
            x: NumPy array, shape = (N, obs_dim)

        see: https://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-
               variance-of-two-groups-given-known-group-variances-mean
        """
        if self.first_pass:
            self.means = np.mean(x, axis=0)
            self.vars = np.var(x, axis=0)
            self.m = x.shape[0]
            self.first_pass = False
        else:
            n = x.shape[0]
            new_data_var = np.var(x, axis=0)
            new_data_mean = np.mean(x, axis=0)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = ((self.means * self.m) + (new_data_mean * n)) / (self.m + n)
            self.vars = (((self.m * (self.vars + np.square(self.means))) +
                          (n * (new_data_var + new_data_mean_sq))) / (self.m + n) -
                         np.square(new_means))
            self.vars = np.maximum(0.0, self.vars)  # occasionally goes negative, clip
            self.means = new_means
            self.m += n

    def get(self):
        """ returns 2-tuple: (scale, offset) """
        return 1/(np.sqrt(self.vars) + 0.1)/3, self.means

class FinalModel:
    def __init__(self, env):
        # Load your Model here
        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph('policy_model/2020710425_model.meta')
        self.saver.restore(self.sess, tf.train.latest_checkpoint('policy_model/'))
        self.action_size = env.action_space.shape[0]
        self.policy = Policy(env.observation_space.shape[0] + 1, self.action_size, 0.003, 10, -1.0, None)



    def get_action(self, state):

        # scale, offset = Scaler.get()
        # scale[-1] = 1.0
        # offset[-1] = 0.0
        # change to your model
        obs = state.astype(np.float32).reshape((1, -1))
        obs = np.append(obs, [[0]], axis=1)  # add time step feature

        action = self.policy.sample(obs).reshape((1, -1)).astype(np.float32)
        return np.squeeze(action, axis=0)
