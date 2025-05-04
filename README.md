# ECE_7202_RL_Project

# 🧠 Offline Reinforcement Learning: Algorithm Survey on D4RL Benchmarks

This repository contains the code and results for our course project on **Offline Reinforcement Learning (Offline RL)**. We perform a comparative study of four offline RL algorithms across standard environments from the [D4RL](https://github.com/rail-berkeley/d4rl) benchmark suite. We use the [d3rlpy](https://d3rlpy.readthedocs.io/en/v2.8.1/), which is an offline deep reinforcement learning library, to implement different algorithms. 

## 📌 Project Overview

Offline RL aims to learn policies from fixed datasets without further environment interaction. In this project, we analyze and compare the performance of the following algorithms:

- **CQL (Conservative Q-Learning)**
- **BEAR (Bootstrapping Error Accumulation Reduction)**
- **BCQ (Batch-Constrained Q-Learning)**
- **AWAC (Advantage-Weighted Actor-Critic)**

## 🧪 Datasets and Environments

We use the following continuous control tasks from the D4RL benchmark:

- `inverted-pendulum` [link](https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/)
- `inverted-double-pendulum` [link](https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/)
- `hopper` [link](https://gymnasium.farama.org/environments/mujoco/hopper/)

For each of the three tasks, we used `medium` and `expert` versions of datasets that are hosted on [Minari](https://minari.farama.org/) where `medium` and `expert` denotes data collected by agents at different levels of training. For example, for `inverted-pendulum`, the `medium` dataset is collected by an agent trained with the [Stable Baseline 3's](https://stable-baselines3.readthedocs.io/en/master/) implementation of Soft-Actor Critic (SAC) for $10\times10^3$ steps while `expert` dataset is collected for agent trained with SAC for $10^6$ steps.


## 🗂️ Repository Structure

- `Pendulum` - code for `inverted-pendulum` task
- `DoublePendulum` - code for `inverted-double-pendulum` task
- `Hopper` - code for the `hopper` task
- dataset_conversion.ipynb - Jupyter Notebook to download dataset from Minari, convert the dataset for d3rlpy and save for further use.
- rendering.ipynb - Jupyter Notebook to render and save a video of a trained agent performing the task.

## Using Minari and d3rlpy

1. Install Minari 
```python 
!pip install "minari[all]"
```

2. List all available offline RL datasets
```python
!minari list remote
```

3. Install Gymnasium Robotics to build Mujoco Environments
```python
!pip install gymnasium_robotics
```

4. Install d3rlpy
```python
!pip install d3rlpy
```


