import random
import os

import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers as ko

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def start(mode, to_recv, to_send):


    while True:
        obs = to_recv.get()
        if obs == None:
            break
        action = random.randint(0, 2)
        to_send.put(action)