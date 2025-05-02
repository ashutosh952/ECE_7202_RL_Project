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
SEED = 1
# BATCH_SIZE = 256
N_STEPS = 200000

# create environment
env = gym.make('Hopper-v5')
# Create the wrapped environment
# wrapped_env = EnvWrapper(env)

d3rlpy.seed(SEED)
d3rlpy.envs.seed_env(env, SEED)


# Load the modified dataset
dataset_name = 'hopper_medium_v0'
data_filename = '../' + dataset_name + '.pkl'

experiment_name = 'bear_'+ dataset_name + '_default_' + str(N_STEPS) + '_seed_' + str(SEED)

with open(data_filename, 'rb') as f:
    dataset = pk.load(f)

vae_encoder = d3rlpy.models.encoders.VectorEncoderFactory([750, 750])

if "halfcheetah" in dataset_name:
    kernel = "gaussian"
else:
    kernel = "laplacian"

bear = BEARConfig(
    actor_learning_rate=1e-4,
    critic_learning_rate=3e-4,
    imitator_learning_rate=3e-4,
    alpha_learning_rate=1e-3,
    imitator_encoder_factory=vae_encoder,
    temp_learning_rate=0.0,
    initial_temperature=1e-20,
    batch_size=256,
    mmd_sigma=20.0,
    mmd_kernel=kernel,
    n_mmd_action_samples=4,
    alpha_threshold=0.05,
    n_target_samples=10,
    n_action_samples=100,
    warmup_steps=40000
).create(device='cuda:2')


bear.fit(
    dataset=dataset,
    n_steps = N_STEPS,           # total number of gradient updates
    n_steps_per_epoch = 1000,  # 3 epochs total
    save_interval = 10,           # save every 1 epoch
    experiment_name = experiment_name,
    with_timestamp=True,
    show_progress=False,
    evaluators={
        "environment": EnvironmentEvaluator(env),
    }
)
