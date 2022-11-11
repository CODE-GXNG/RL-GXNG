# ReinforcementLearning-Minihack-QuestHard-v0
This repository is the source code for creating an agent to complete the Minihack-Quest-Hard-v0 environment. 

We provide two different algorithm implementations:
1. A deep NN policy gradient model trained with PPO
2. A value based deep Q NN trained with Neuro-evolution.
   
## Scripts
The scripts folder has a series of python files available for different tasks:

### PPO.py 
An executable file used to carry out the PPO algorithm from the stable_baselines3 package.

Edit the following variables to configure your run: {MODEL_SAVING_DIR,MODEL_NAME,tmp_path} 

The scores data will be saved to the tmp_path. The model will have its checkpoints saved with the name MODEL_NAME to the MODEL_SAVING_DIR

### NE.py 
An An executable file used to implement a the Value based NN optimised with the NE algorithm, using both GA and PSO as optimisiers.

The model is saved to output.zip. It can be loaded with the NN.load_model()

### environment.py
A module that includes all methods linked to the gym environment. 

Specifically, it has four methods: 
1. get_actions()
1. get_reward manager(obskeys)
2. get_obs_keys(pixel,message)
1. init_env(pixel=False,customreward=True,message=False). 
The get actions() function produces a
2-tuple that includes a tuple of the agent’s accessible utility actions and a
tuple of potential positional actions. The reward system that the agent will
utilize as feedback while learning is described in the get reward manager(obs
keys) function. Depending on whether the pixel and message arguments
are true or false, the get obs keys(pixel,message) function provides
a tuple of observation spaces that might contain or omit the pixel and
message observation space. If the ”custom reward” keyword is true, the
init env(pixel=False,custom reward=True,message=False) function
produces a gym environment with a custom reward manager; otherwise,
it returns a gym environment with the default reward manager.

### utilities.py 
A module that includes all methods for displaying an agent’s performance in its surroundings.

It has three methods: 
1. frames_to_gif(frames)
1. generate_video(model)
1. save_gif(gif,path). 

A list of image objects or gifs is sent to the save gif(gif,path) function, which stores the data at the given path. Using a list of instances of pixel observations as input, the frames_to_gif(frames) function creates a gif using the provided instances of pixel observations. The generate_video(model) function calls on the above methods to generate a gif.

## Requirements
The requirements.txt file has the list of requirements to run these scripts.

