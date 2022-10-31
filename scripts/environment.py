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

def init_env():
    device = torch.device("cpu")
    # obs_keys = ('glyphs', 'chars','screen_descriptions', 'glyphs_crop', 'chars_crop', 'screen_descriptions_crop','blstats', 'message', 'inv_strs', 'pixel')
    obs_keys = ('glyphs','glyphs_crop','blstats','message','inv_glyphs', 'pixel')

    reward_manager = RewardManager()
    reward_manager.add_eat_event("apple", reward=1)
    reward_manager.add_wield_event("dagger", reward=2)
    reward_manager.add_wield_event("wand", reward=2)
    reward_manager.add_location_event("sink", reward=-1, terminal_required=False)
    reward_manager.add_amulet_event()
    reward_manager.add_kill_event("demon")
    env = gym.make(
        "MiniHack-Quest-Hard-v0",
        observation_keys=obs_keys,
        reward_manager=reward_manager
    )
    return env