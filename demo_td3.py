import sys

import numpy as np
import matplotlib.pyplot as plt

from torch import tensor
from torch import nn
import torch.nn.functional as F
import torch

import gym

import pomdp

rand_seed = 42

environment_name = sys.argv[1]
pomdp_type = sys.argv[2]
model_path = sys.argv[3]
num_eval_episodes = int(sys.argv[4])

parameters = {
    'num_iterations': {
        'Ant-v2': 1_000_000,
        'Hopper-v2': 1_000_000
    },
    'replay_memory_max_size': {
        'Ant-v2': 1_000_000,
        'Hopper-v2': 1_000_000
    },
    'sigma': {
        'Ant-v2': 0.2,
        'Hopper-v2': 0.2
    },
    'minibatch_size': {
        'Ant-v2': 100,
        'Hopper-v2': 100
    },
    'discount_rate': {
        'Ant-v2': 0.99,
        'Hopper-v2': 0.99
    },
    'steps_until_policy_update': {
        'Ant-v2': 2,
        'Hopper-v2': 2
    },
    'target_update_ratio': {
        'Ant-v2': 0.0005,
        'Hopper-v2': 0.0005
    },
    'epsilon_limit': {
        'Ant-v2': 0.5,
        'Hopper-v2': 0.5
    },
    'smoothing_sigma': {
        'Ant-v2': 0.1,
        'Hopper-v2': 0.1
    },
    'num_iter_before_train': {
        'Ant-v2': 10_000,
        'Hopper-v2': 1000
    },
    'state_dim': {
        'Ant-v2': 111,
        'Hopper-v2': 11
    },
    'action_dim': {
        'Ant-v2': 8,
        'Hopper-v2': 3
    }
}

state_dim = parameters['state_dim'][environment_name]
action_dim = parameters['action_dim'][environment_name]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu_device = torch.device("cpu")


class TD3LunarLanderContinuousActorNN(nn.Module):
    def __init__(self):
        super(TD3LunarLanderContinuousActorNN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


env = gym.make(environment_name)

if pomdp_type == 'faulty':
    if environment_name == 'Hopper-v2':
        po_env = pomdp.PartiallyObservableEnv(env, rand_seed, faulty=(20, env.observation_space.shape[0]))
    elif environment_name == 'Ant-v2':
        po_env = pomdp.PartiallyObservableEnv(env, rand_seed, faulty=(20, 27))
elif pomdp_type == 'noisy':
    po_env = pomdp.PartiallyObservableEnv(env, rand_seed, noisy=(0, 0.1))
else:
    po_env = pomdp.PartiallyObservableEnv(env, rand_seed)

actor_target_net = TD3LunarLanderContinuousActorNN().to(device)
actor_target_net.load_state_dict(torch.load(model_path))

render = True

obs = po_env.reset()
episode_rewards = []
episode_reward = 0
while len(episode_rewards) < num_eval_episodes:
    action = actor_target_net(tensor(obs).float().to(device)).detach().to(cpu_device).numpy().squeeze()
    obs, reward, done, _ = po_env.step(action)
    episode_reward += reward
    if done:
        episode_rewards.append(episode_reward)
        episode_reward = 0
        obs = po_env.reset()
    if render:
        po_env.render()

episode_rewards = np.asarray(episode_rewards)
episode_length_histogram = plt.hist(episode_rewards)
plt.title("Episode Rewards")
plt.xlabel("Total Reward")
plt.ylabel("Frequency")
print("Mean total episode reward:", np.mean(episode_rewards))