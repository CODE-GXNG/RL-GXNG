import gym
import minihack
from stable_baselines3 import DQN
from environment import init_env
from utilities import generate_video
from stable_baselines3.common.callbacks import BaseCallback,CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

        

def main():
    '''
    https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
    '''
    MODEL_SAVING_DIR = "../model_saving_dir/DQN/"
    MODEL_NAME = "dqn"
    MODEL_SAVE_PATH = MODEL_SAVING_DIR+MODEL_NAME
    log_dir = '../logs/'
    env = init_env()
    env = Monitor(env, log_dir)
    model = DQN("MultiInputPolicy", env, verbose=1)

    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=MODEL_SAVING_DIR)
    # Separate evaluation env
    eval_env = init_env()
    eval_callback = EvalCallback(eval_env, best_model_save_path=MODEL_SAVING_DIR+"best_model",
                                log_path="../logs/results", eval_freq=500)

    callback = CallbackList([checkpoint_callback, eval_callback])

    model.save(MODEL_SAVE_PATH)
    for i in range(100):
        print("Starting epoch {}".format(i))
        model.learn(total_timesteps=1000000, log_interval=4)
        print("Finished epoch {}".format(i))
        model.save(MODEL_SAVE_PATH)
        if i%10==0: 
            model.save(MODEL_SAVE_PATH+"{}_timesteps".format(i))
            generate_video(model,title='training_{}'.format(i))


        print("Saved model checkpoints")
    model.save(MODEL_SAVE_PATH)
    print("Done")



if __name__ == "__main__":
    main()