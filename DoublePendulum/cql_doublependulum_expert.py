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
BATCH_SIZE = 1024
N_STEPS = 100000

# create environment
env = gym.make('InvertedDoublePendulum-v5')
# Create the wrapped environment
# wrapped_env = EnvWrapper(env)

d3rlpy.seed(SEED)
d3rlpy.envs.seed_env(env, SEED)


# Load the modified dataset
dataset_name = 'doublependulum_expert_v0'
data_filename = dataset_name + '.pkl'

experiment_name = 'cql_'+ dataset_name + '_' + str(BATCH_SIZE) + '_' + str(N_STEPS) + '_seed_' + str(SEED)

with open(data_filename, 'rb') as f:
    dataset = pk.load(f)

# encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])

# if "medium_v0" in dataset_name:
#     conservative_weight = 10.0
# else:
#     conservative_weight = 5.0

# cql = CQLConfig(
#     actor_learning_rate=1e-4,
#     critic_learning_rate=3e-4,
#     temp_learning_rate=1e-4,
#     actor_encoder_factory=encoder,
#     critic_encoder_factory=encoder,
#     batch_size=256,
#     n_action_samples=10,
#     alpha_learning_rate=0.0,
#     conservative_weight=conservative_weight
# ).create(device='cuda:0')

cql = CQLConfig().create('cuda:0')

cql.fit(
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
