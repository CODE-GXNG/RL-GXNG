import gym
import minihack
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
from collections import deque
import time
from datetime import datetime
import random
from minihack import RewardManager
from stable_baselines3 import DQN
from environment import init_env

def main():
    env = init_env()
    model = DQN("MlpPolicy", env, verbose=1)
    model.save("dqn")
    for i in range(10):
        model.learn(total_timesteps=1000, log_interval=4)
        model.save("dqn")

    del model # remove to demonstrate saving and loading

    model = DQN.load("dqn")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    
if __name__ == "__main__":
    main()