from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
import bsp
import pandas as pd

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

variable = ['pH Start', 'j [mA/cm2]', 'flow rate [l/min]', 't [min]']
max_variable = [1.0, 1.0, 1.0, 1.0]

filename = "knn_flow_CE-S-yield.sav"
reactants_dict = zip(variable, max_variable)
parameter = [len(max_variable)]
max_episodes = 10000
maxsize = 5

# load environment
env = bsp.Parameteroptimierung(
    filename=filename, max_variable=max_variable,
    max_episodes=max_episodes, maxsize=maxsize)

check_env(env) # check environment


# PPO
# policy is MLPPolicy - this is predefined
model = PPO("MlpPolicy", env, verbose=0, device='cuda', tensorboard_log='/home/sc.uni-leipzig.de/pr481jlua/ML4AUTO/RFL_Antonia/V.0.9/mylog',batch_size=256, learning_rate=5e-4)

# checkpoint model save
checkpoint_callback = CheckpointCallback(2000, "/home/sc.uni-leipzig.de/pr481jlua/RL/modelcheck/models")


model.learn(total_timesteps=1000000, progress_bar=False,tb_log_name="model_flow_CE-S-yield")
# callback=checkpoint_callback)

model.save("/home/sc.uni-leipzig.de/pr481jlua/RL/model/mmodel_flow_CE-S-yield")

