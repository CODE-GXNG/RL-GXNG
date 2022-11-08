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
        # nethack.Command.APPLY,
        # nethack.Command.FIRE,
        nethack.Command.RUSH,
        nethack.Command.ZAP,
        # nethack.Command.PUTON, #armour
        # nethack.Command.READ, # a scroll or spellbook
        # nethack.Command.WEAR,
        nethack.Command.QUAFF, #drink
        # nethack.Command.PRAY,
        nethack.Command.WIELD,
        nethack.Command.OPEN ,
        # nethack.Command.FORCE ,
        # nethack.Command.KICK ,
        # nethack.Command.LOOT ,
    )
    return MOVE_ACTIONS + NAVIGATE_ACTIONS

def get_reward_manager(obs_keys):
    observation_keys = list(obs_keys)
    glyph_index = observation_keys.index("glyphs")

    reward_manager = RewardManager()
    reward_manager.add_eat_event("apple", reward=1,terminal_required=False,terminal_sufficient=False)
    # reward_manager.add_wield_event("dagger", reward=2)
    reward_manager.add_wield_event("wand", reward=2,terminal_required=False)
    reward_manager.add_location_event("sink", reward=-1, terminal_required=False)
    reward_manager.add_amulet_event(reward=1,terminal_required=False)
    reward_manager.add_kill_event("demon",reward=1,terminal_required=False)
    reward_manager.add_kill_event("monster",reward=1,terminal_required=False)

    # def reward_fn(env, prev_obs, action, next_obs):
    #     reward = 0.0
    #     return reward
    def explore_reward_fn(env,last_observation, action, observation):
        '''
        https://github.com/facebookresearch/nle/blob/0c5d66f8902929ba3963d38780a23ff79b72e7e8/nle/env/tasks.py
        '''
        obs_keys = env.observation_space.keys()
        observation_keys = list(obs_keys)
        glyph_index = observation_keys.index("glyphs")

        glyphs = observation[glyph_index]
        last_glyphs = last_observation[glyph_index]

        explored = np.sum(glyphs != nethack.GLYPH_CMAP_OFF)
        explored_old = np.sum(last_glyphs != nethack.GLYPH_CMAP_OFF)

        reward = explored - explored_old

        return float(reward )


    # reward_manager.add_custom_reward_fn(explore_reward_fn)

    reward_manager.add_message_event(
        ["f - a copper wand.","f - a silver wand","g - a copper wand.","g - a silver wand","f - a uranium wand.","g - a uranium wand"],
        reward=1,
        repeatable=True,
        terminal_required=False,
        terminal_sufficient=False,
    )

    reward_manager.add_message_event(
        ["The door opens.","The lava cools and solidifies."],
        reward=2,
        repeatable=True,
        terminal_required=False,
        terminal_sufficient=False,
    )

    reward_manager.add_kill_event("minotaur",reward=2,terminal_required=True)


    return reward_manager

def get_obs_keys(pixel,message):
    '''
        Possible keys: ('glyphs', 'chars','screen_descriptions', 'glyphs_crop', 'chars_crop', 'screen_descriptions_crop','blstats', 'message', 'inv_strs', 'pixel')

    '''

    if pixel and message:   return  ('glyphs','glyphs_crop','blstats','inv_glyphs', 'pixel','message')
    elif message:           return  ('glyphs','glyphs_crop','blstats','inv_glyphs','message')
    elif pixel:             return  ('glyphs','glyphs_crop','blstats','inv_glyphs','pixel')
    else:                   return  ('glyphs','glyphs_crop','blstats','inv_glyphs')

def init_env(pixel=False,custom_reward=True,message=False):
    obs_keys = get_obs_keys(pixel,message)
    if custom_reward:
        return gym.make(
            "MiniHack-Quest-Hard-v0",
            reward_win=5,
            reward_lose=-2,
            observation_keys=obs_keys,
            reward_manager=get_reward_manager(obs_keys),
            actions=get_actions()
        )
    else:
        return gym.make(
            "MiniHack-Quest-Hard-v0",
            observation_keys=obs_keys,
            actions=get_actions()
        )
