# Required Imports
import gym
import minihack
from stable_baselines3 import PPO
from environment import init_env
from utilities import generate_video
from stable_baselines3.common.callbacks import BaseCallback,CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor, ResultsWriter
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
        

def main():
    '''
    https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
    '''
    MODEL_SAVING_DIR = "../model_saving_dir/PPO/"
    MODEL_NAME = "ppo"
    MODEL_SAVE_PATH = MODEL_SAVING_DIR+MODEL_NAME
    log_path = "../logs/PPO_log/"

    # Parallel environments
    # Parallel environments allows PPO to run multiple agents pairs at the same time. This will increase the speed at which environment steps are generated, and will make transitions within a batch more IID too.
    # Make_vec_env is method for stacking multiple independent environments into a single environment, and use init_env from environment as import for the method
    env = make_vec_env(init_env, n_envs=4)

    model = PPO("MultiInputPolicy", env, verbose=1)
    
    ## Checkpoint CallBack -
    ## Callback for saving a model every save freq 
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=MODEL_SAVE_PATH)
        
    ## EvalCallback 
    # Separate evaluation env
    eval_env = init_env()

    ## evaluate periodically the performance of an agent, it will save the best model 
    eval_callback = EvalCallback(eval_env, best_model_save_path=MODEL_SAVING_DIR+"best_model", log_path="../logs/PPO/results", eval_freq=500)
    callback = CallbackList([checkpoint_callback, eval_callback])

    #setting up configure used For getting the results needed for the reward per episode grapgs
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # PPO models trains
    model.learn(total_timesteps=100000000,callback=callback)
    model.save()


if __name__ == "__main__":
    main()
