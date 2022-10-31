import gym
import minihack
from stable_baselines3 import DQN
from environment import init_env
from utilities import generate_video

        

def main():
    '''
    https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
    '''
    MODEL_SAVING_DIR = "../model_saving_dir/"
    MODEL_NAME = "dqn"
    MODEL_SAVE_PATH = MODEL_SAVING_DIR+MODEL_NAME

    env = init_env()
    model = DQN("MultiInputPolicy", env, verbose=1)

    model.save(MODEL_SAVE_PATH)
    for i in range(10):
        print("Starting epoch {}".fomrat(i))
        model.learn(total_timesteps=10, log_interval=4)
        print("Finished epoch {}".fomrat(i))
        model.save(MODEL_SAVE_PATH)
        model.save(MODEL_SAVE_PATH+"{}_timesteps".format(i))
        print("Saved model checkpoints")
        generate_video(model,title='training_{}'.format(i))
    model.save(MODEL_SAVE_PATH)
    print("Done")


    
if __name__ == "__main__":
    main()