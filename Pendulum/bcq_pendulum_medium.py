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

experiment_name = 'bcq_'+ dataset_name + '_' + str(BATCH_SIZE) + '_' + str(N_STEPS) + '_seed_' + str(SEED)

with open(data_filename, 'rb') as f:
    dataset = pk.load(f)

# vae_encoder = d3rlpy.models.encoders.VectorEncoderFactory([750, 750])
# rl_encoder = d3rlpy.models.encoders.VectorEncoderFactory([400, 300])

# bcq = BCQConfig(
#     actor_encoder_factory=rl_encoder,
#     actor_learning_rate=1e-3,
#     critic_encoder_factory=rl_encoder,
#     critic_learning_rate=1e-3,
#     imitator_encoder_factory=vae_encoder,
#     imitator_learning_rate=1e-3,
#     batch_size=100,
#     lam=0.75,
#     action_flexibility=0.05,
#     n_action_samples=100
# ).create(device='cuda:1')

bcq = BCQConfig(batch_size = BATCH_SIZE).create('cuda:1')

bcq.fit(
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
