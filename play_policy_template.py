import sys
from model import CarActionModel
import numpy as np
from torch import unsqueeze, detach
from torchvision import transforms
from torchvision.io import read_image
from PIL import Image
import torch
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
        #p = model.predict(obs) # adapt to your model
        #print("OLE", type(obs))
        #pil_image = Image.fromarray(obs)  # Assuming values are in the range [0, 1]

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
        ])
        #print(transform(obs).unsqueeze(0))
        image = transform(obs).unsqueeze(0)
        #print(image.shape)
        p = model(image)
        #print(p.shape)

        action = int(np.argmax(p.detach()))  # adapt to your model
        #print("ACTION", action)
        
        obs, _, terminated, truncated, _ = env.step(action)
        #print("TRUNCATED",terminated)
        done = terminated or truncated




env_arguments = {
    'domain_randomize': False,
    'continuous': False,
    'render_mode': 'human',
    "max_episode_steps":3000
}

env_name = 'CarRacing-v2'
env = gym.make(env_name, **env_arguments)

print("Environment:", env_name)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

# your trained
model = CarActionModel.load_from_checkpoint("/home/bruno/Desktop/ML_EX2/Saves/ckpt/0.01, 0.01, 0, 0.5, 0.01, 0.01-v1.ckpt") # your trained model
model.eval()
play(env, model)


##