import gymnasium as gym
import gymnasium_robotics
import minari
import numpy as np
import d3rlpy
from d3rlpy.datasets import MDPDataset

from d3rlpy.datasets import get_d4rl
from d3rlpy.algos import CQLConfig, BEARConfig, AWACConfig, BCQConfig
from d3rlpy.metrics.evaluators import EnvironmentEvaluator
from gym.wrappers import RecordVideo

import pickle as pk

# class EnvWrapper(gym.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)

#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         return obs["observation"], info

#     def step(self, action):
#         obs, reward, terminated, truncated, info = self.env.step(action)
#         return obs["observation"], reward, terminated, truncated, info


# fix seed
SEED = 0
BATCH_SIZE = 4096
N_STEPS = 100000

# create environment
env = gym.make('InvertedPendulum-v5')
# Create the wrapped environment
# wrapped_env = EnvWrapper(env)

d3rlpy.seed(SEED)
d3rlpy.envs.seed_env(env, SEED)


# Load the modified dataset
dataset_name = 'pendulum_medium_v0'
data_filename = dataset_name + '.pkl'

experiment_name = 'awac_'+ dataset_name + '_' + str(BATCH_SIZE) + '_' + str(N_STEPS) + '_seed_' + str(SEED)

with open(data_filename, 'rb') as f:
    dataset = pk.load(f)

# encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256, 256])
# optim = d3rlpy.optimizers.AdamFactory(weight_decay=1e-4)

# awac = AWACConfig(
#         actor_learning_rate=3e-4,
#         actor_encoder_factory=encoder,
#         actor_optim_factory=optim,
#         critic_learning_rate=3e-4,
#         critic_encoder_factory=encoder,
#         batch_size=1024,
#         lam=1.0
#     ).create(device='cuda:1')

awac = AWACConfig(batch_size = BATCH_SIZE).create('cuda:1')

awac.fit(
    dataset=dataset,
    n_steps = N_STEPS,           # total number of gradient updates
    n_steps_per_epoch = 1000,  # 3 epochs total
    save_interval = 1,           # save every 1 epoch
    experiment_name = experiment_name,
    with_timestamp=True,
    show_progress=False,
    evaluators={
        "environment": EnvironmentEvaluator(env),
    }
)
