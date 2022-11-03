import gym
import minihack
from stable_baselines3 import PPO
from environment import init_env
from utilities import generate_video
from stable_baselines3.common.callbacks import BaseCallback,CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np

        

def main():
    '''
    https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
    '''
    MODEL_SAVING_DIR = "../model_saving_dir/DQN/"
    MODEL_NAME = "dqn60_timesteps"
    MODEL_SAVE_PATH = MODEL_SAVING_DIR+MODEL_NAME
    log_dir = '../logs/'
    env = init_env()

    # model = PPO("MultiInputPolicy", env, verbose=1)
    models = []
    seeds = np.random.randint(1000, size=10)

    performance_data = []
    for model_path in models:

        rewards = []
        for seed in seeds:
            env.seed(seed)
            model = PPO("MultiInputPolicy", env, verbose=1)
            model.load(model_path)

            done = False
            obs = env.reset()
            reward=0
            while not done:
                action, _states = model.predict(obs, deterministic=False)
                obs, r, done, info = env.step(action)
                reward+=r
            rewards.append(reward)
        performance_data.append(rewards)
    performance_data = np.array(performance_data).T
    np.savetxt("performance_data.csv", performance_data, delimiter=",")




if __name__ == "__main__":
    main()
