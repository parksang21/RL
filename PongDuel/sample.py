import random
import os

import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers as ko

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__(name='dqn')

        self.shared_fc1 = kl.Dense(16, activation='relu', kernel_initializer='he_uniform')
        self.shared_fc2 = kl.Dense(32, activation='relu', kernel_initializer='he_uniform')

        self.val_adv_fc = kl.Dense(num_actions+1, activation='relu', kernel_initializer='he_uniform')

    def call(self, inputs):
        x = self.shared_fc1(inputs)
        x = self.shared_fc2(x)
        val_adv = self.val_adv_fc(x)

        outputs = tf.expand_dims(val_adv[:, 0], -1) + (val_adv[:, 1:]-tf.reduce_mean(val_adv[:,1:], -1, keepdims=True))
        return outputs

    def action_value(self, obs):
        q_values = self.predict(obs)
        best_action = np.argmax(q_values, axis=-1)
        return best_action if best_action.shape[0] > 1 else best_action[0], q_values[0]

def start(mode, to_recv, to_send):


    while True:
        obs = to_recv.get()
        if obs == None:
            break
        action = random.randint(0, 2)
        to_send.put(action)