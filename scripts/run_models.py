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
    env = init_env(message=True)

    # model = PPO("MultiInputPolicy", env, verbose=1)
    models = ["model_saving_dir\DQN\dqn0_timesteps",""]
    seeds = np.random.randint(1000, size=10)

    performance_data = []
    subgoal_data = []
    for model_path in models:
        subgoal_counts = np.zeros(3)
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
                messageUINT8 = obs["message"]
                messageString = "".join([chr(item)  for item in messageUINT8 if chr(item) != "\x00"])
                if "The door opens." in messageString:
                    subgoal_counts[0]+=1
                if "The lava cools and solidifies." in messageString:
                    subgoal_counts[1]+=1
                if "You kill the minotaur!" in messageString:
                    subgoal_counts[2]+=1
                reward+=r
            rewards.append(reward)
        performance_data.append(rewards)
        subgoal_counts/=len(seeds)
        subgoal_data.append(subgoal_counts)

    performance_data = np.array(performance_data).T
    subgoal_data = np.array(subgoal_data).T
    np.savetxt("performance_data.csv", performance_data, delimiter=",")
    np.savetxt("subgoal_data.csv", subgoal_data, delimiter=",")




if __name__ == "__main__":
    main()
