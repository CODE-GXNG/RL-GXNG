import gym
import minihack
from nle import nethack
from minihack import RewardManager
import numpy as np


def get_actions():
    '''
    https://nethackwiki.com/wiki/Commands
    https://github.com/facebookresearch/nle/blob/091f1c18a10a4b4c497f8adda046794081f78105/nle/nethack/actions.py
    '''
    MOVE_ACTIONS = tuple(nethack.CompassDirection)
    NAVIGATE_ACTIONS =  (
        nethack.Command.PICKUP,
        nethack.Command.APPLY,
        nethack.Command.FIRE,
        nethack.Command.RUSH,
        nethack.Command.ZAP,
        nethack.Command.PUTON, #armour
        nethack.Command.READ, # a scroll or spellbook
        nethack.Command.WEAR,
        nethack.Command.QUAFF, #drink
        nethack.Command.PRAY,
        nethack.Command.WIELD,
        nethack.Command.OPEN ,
        nethack.Command.FORCE ,
        nethack.Command.KICK ,
        nethack.Command.LOOT ,
    )
    return MOVE_ACTIONS + NAVIGATE_ACTIONS

def get_reward_manager():


    reward_manager = RewardManager()
    reward_manager.add_eat_event("apple", reward=1)
    # reward_manager.add_wield_event("dagger", reward=2)
    reward_manager.add_wield_event("wand", reward=2)
    reward_manager.add_location_event("sink", reward=-1, terminal_required=False)
    reward_manager.add_amulet_event()
    reward_manager.add_kill_event("demon")

    def reward_fn(env, prev_obs, action, next_obs):
        reward = 0.0
        return reward
    def scout_reward_fn(last_observation, action, observation, end_status):
        '''
        https://github.com/facebookresearch/nle/blob/0c5d66f8902929ba3963d38780a23ff79b72e7e8/nle/env/tasks.py
        '''
        glyphs = observation['glyphs']
        last_glyphs = last_observation['glyphs']

        explored = np.sum(glyphs != nethack.GLYPH_CMAP_OFF)
        explored_old = np.sum(last_glyphs != nethack.GLYPH_CMAP_OFF)

        reward = explored - explored_old

        return reward 

    # reward_manager.add_custom_reward_fn(scout_reward_fn)

    return reward_manager

def get_obs_keys(pixel):
    '''
        Possible keys: ('glyphs', 'chars','screen_descriptions', 'glyphs_crop', 'chars_crop', 'screen_descriptions_crop','blstats', 'message', 'inv_strs', 'pixel')

    '''
    if pixel: return  ('glyphs','glyphs_crop','blstats','inv_glyphs', 'pixel')
    else:return  ('glyphs','glyphs_crop','blstats','inv_glyphs')

def init_env(pixel=False):
    env = gym.make(
        "MiniHack-Quest-Hard-v0",
        observation_keys=get_obs_keys(pixel),
        reward_manager=get_reward_manager(),
        actions=get_actions()
    )
    return env