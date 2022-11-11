import gym
import minihack
from stable_baselines3 import PPO
from environment import init_env
from utilities import generate_video
from stable_baselines3.common.callbacks import BaseCallback,CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor, ResultsWriter
from stable_baselines3.common.logger import configure

# Changes Here
from stable_baselines3.common.env_util import make_vec_env
        

def main():
    '''
    https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
    '''


    env = init_env()
    model = PPO("MultiInputPolicy", env, verbose=1)
    
    ## Continue With The Prev Model
    PREV_MODEL_PATH = ""#TODO
    model.load(PREV_MODEL_PATH)

    for i in range(100):
    	generate_video(model,title='PPO_GIFS/PPO_GIF_{}'.format(i))

if __name__ == "__main__":
    main()
