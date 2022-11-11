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
    MODEL_SAVING_DIR = "../model_saving_dir/PPO/"
    MODEL_NAME = "ppo"
    MODEL_SAVE_PATH = MODEL_SAVING_DIR+MODEL_NAME
    log_path = "../logs/PPO_log/"

    # Changes Here (Added Parallel Environments)
    # Parallel environments
    # Parallel environments are just an optimization trick that allows PPO to run multiple agent/environment pairs at the same time. This can greatly increase the speed at which environment steps are generated, and can also make the transitions within a batch more IID too.
    env = make_vec_env(init_env, n_envs=4)

    model = PPO("MultiInputPolicy", env, verbose=1)
    
    ## Continue With The Prev Model, Unquote below lines to load prev model
    # PATH_TO_PREV_MODEL = TODO 
    # model.load(PATH_TO_PREV_MODEL)
    # model.set_env(env)
    
    ## Checkpoint CallBack -
    ## Callback for saving a model every save freq 
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=MODEL_SAVE_PATH)
    ## EvalCallback 
    # Separate evaluation env
    eval_env = init_env()
    ## evaluate periodically the performance of an agent, it will save the best model 
    eval_callback = EvalCallback(eval_env, best_model_save_path=MODEL_SAVING_DIR+"best_model", log_path="../logs/PPO/results", eval_freq=500)
    callback = CallbackList([checkpoint_callback, eval_callback])

    #Setting Up Configure
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    
    model.learn(total_timesteps=100000000,callback=callback)
    model.save()


if __name__ == "__main__":
    main()
