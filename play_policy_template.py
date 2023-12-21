import sys
from model import CarActionModel
import numpy as np
from torch import unsqueeze, detach
from torchvision.io import read_image
from PIL import Image
import utils
import torch
import os
try:
    import gymnasium as gym
except ModuleNotFoundError:
    print('gymnasium module not found. Try to install with')
    print('pip install gymnasium[box2d]')
    sys.exit(1)


def play(env, model):

    seed = 2000
    obs, _ = env.reset(seed=seed)
    
    # drop initial frames
    action0 = 0
    for i in range(50):
        obs,_,_,_,_ = env.step(action0)
    
    done = False
    while not done:
        p = model.predict(obs) # adapt to your model     
        obs, _, terminated, truncated, _ = env.step(p)
        #print("TRUNCATED",terminated)
        done = terminated or truncated




env_arguments = {
    'domain_randomize': False,
    'continuous': False,
    'render_mode': 'human',
    #"max_episode_steps":3000
}

env_name = 'CarRacing-v2'
env = gym.make(env_name, **env_arguments)

print("Environment:", env_name)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

# your trained
model = CarActionModel.load_from_checkpoint(os.path.join(utils.CKPT_SAVE_DIR_NAME,"0.01, 0, 0.5, 0.01, 0, 0.5.ckpt"), map_location='cpu') # your trained model
model.eval()
play(env, model)


##