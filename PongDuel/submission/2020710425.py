import random
import os

import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers as ko

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class Model(tf.keras.Model):
    def __init__(self, model_num):
        super().__init__(name='basic_ddqn{}'.format(model_num))
        # you can try different kernel initializer
        self.fc1 = kl.Dense(32, activation='relu', kernel_initializer='he_uniform')
        self.fc2 = kl.Dense(32, activation='relu', kernel_initializer='he_uniform')
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


def start(mode, to_recv, to_send):
    model = Model(1)
    if mode == "left":
        model.load_weights("./2020710425_model.left")
    else:
        model.load_weights("./2020710425_model.right")

    obs_window = np.zeros((4 * 12,))
    while True:
        obs = to_recv.get()

        if obs == None:
            break

        for i in range(12, 3 * 12 + 1, 12):
            obs_window[i - 12:i] = obs_window[i:i + 12]
        obs = np.asarray(obs[0] + obs[1][:2])
        obs_window[-12:] = obs
        q_values = model.predict(obs_window[None])
        best_action = np.argmax(q_values, axis=-1)
        # action = random.randint(0, 2)
        to_send.put(best_action)
