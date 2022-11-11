from PIL import Image
import gym
import minihack
from datetime import datetime
from environment import init_env


def save_gif(gif,path):
    '''
    Args:
        gif: a list of image objects
        path: the path to save the gif to
    '''
    path=path+'.gif'
    gif[0].save(path, save_all=True,optimize=False, append_images=gif[1:], loop=0)
    print("Saved Video")

def frames_to_gif(frames):
    '''
    Converts a list of pixel value arrays to a gif list of images
    '''
    gif = []
    for image in frames:
        gif.append(Image.fromarray(image, "RGB"))
    return gif


def generate_video(model,title=None,path='../videos/'):
    '''
    Generates a gif for a model, saves it
    Args:
        model - a class with a predict method that takes in a state observation and returns an action
        title - the title of the gif 
        the path to save the gif to
    Ret:
        None
    '''
    frames = []
    env = init_env(pixel=True,custom_reward=True)
    done = False
    obs = env.reset()
    while not done:
        del obs['pixel']
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        frames.append(obs["pixel"])
        

    if title is None:
        title = datetime.now().strftime("%d-%m-%Y_%H:%M:%S") 

    gif = frames_to_gif(frames)
    save_gif(gif,path+title)