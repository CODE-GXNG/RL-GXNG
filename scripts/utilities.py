from PIL import Image
import gym
import minihack
from datetime import datetime


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
    gif = []
    for image in frames:
        gif.append(Image.fromarray(image, "RGB"))
    return gif


def generate_video(model,env,title=None,path='../videos/'):
    frames = []
    
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        frames.append(obs["pixel"])

    if title is None:
        title = datetime.now().strftime("%d-%m-%Y_%H:%M:%S") 

    gif = frames_to_gif(frames,path+title)
    save_gif(gif,path)