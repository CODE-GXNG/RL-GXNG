import gym
import minihack
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import deque
import time
from datetime import datetime
import random
from stable_baselines3 import DQN
from environment import init_env
from utilities import generate_video

        

def main():
    '''
    https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
    '''
    env = init_env()
    model = DQN("MlpPolicy", env, verbose=1)
    model.save("../model_saving-dir/dqn")
    for i in range(10):
        model.learn(total_timesteps=10, log_interval=4)
        model.save("../model_saving-dir/dqn")
        model.save("../model_saving-dir/dqn_{}_timesteps".format(i))
        generate_video(model,env,title='training_{}'.format(i))
    model.save("../model_saving-dir/dqn")
    
if __name__ == "__main__":
    main()