import sys

import numpy as np
import matplotlib.pyplot as plt

from torch import tensor
from torch import clamp
from torch.distributions import normal
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
        'Ant-v2': 3_000_000,
        'Hopper-v2': 1_000_000
    },
    'learning_rate': {
        'Ant-v2': 0.0003,
        'Hopper-v2': 0.0003
    },
    'discount_rate': {
        'Ant-v2': 0.99,
        'Hopper-v2': 0.99
    },
    'replay_buffer_max_size': {
        'Ant-v2': 1_000_000,
        'Hopper-v2': 1_000_000
    },
    'target_smoothing_coefficient': {
        'Ant-v2': 0.005,
        'Hopper-v2': 0.005
    },
    'target_update_interval': {
        'Ant-v2': 1,
        'Hopper-v2': 1
    },
    'num_gradient_steps': {
        'Ant-v2': 1,
        'Hopper-v2': 1
    },
    'num_env_steps': {
        'Ant-v2': 1,
        'Hopper-v2': 1
    },
    'reward_scale': {
        'Ant-v2': 5,
        'Hopper-v2': 5
    },
    'minibatch_size': {
        'Ant-v2': 256,
        'Hopper-v2': 256
    },
    'state_dim': {
        'Ant-v2': 111,
        'Hopper-v2': 11
    },
    'action_dim': {
        'Ant-v2': 8,
        'Hopper-v2': 3
    },
    'num_iter_before_train': {
        'Ant-v2': 4000,
        'Hopper-v2': 4000
    },
    'hidden_layer_size': {
        'Ant-v2': 256,
        'Hopper-v2': 256
    }
}

STATE_DIM = parameters['state_dim'][environment_name]
ACTION_DIM = parameters['action_dim'][environment_name]
hidden_layer_size = parameters['hidden_layer_size'][environment_name]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu_device = torch.device("cpu")

class SACActorNN(nn.Module):
    def __init__(self):
        super(SACActorNN, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.mean = nn.Linear(hidden_layer_size, ACTION_DIM)
        self.log_stdev = nn.Linear(hidden_layer_size, ACTION_DIM)
        self.normal_dist = normal.Normal(0, 1)


    def forward(self, x_state):
        x_state = F.relu(self.fc1(x_state))
        x_state = F.relu(self.fc2(x_state))
        mean = self.mean(x_state)
        log_stdev = clamp(self.log_stdev(x_state), min=-2, max=20)
        unsquashed_action = mean + self.normal_dist.sample(sample_shape=log_stdev.shape).to(device) * torch.exp(log_stdev).to(device)
        squashed_action = torch.tanh(unsquashed_action)
        action_dist = normal.Normal(mean, torch.exp(log_stdev))
        log_prob_squashed_a = action_dist.log_prob(unsquashed_action).to(device) - torch.sum(torch.log(clamp(torch.ones(squashed_action.shape).to(device) - squashed_action**2, min=1e-8)), dim=1).view(-1, 1).repeat(1, ACTION_DIM)
        return squashed_action, log_prob_squashed_a


actor_net = SACActorNN().to(device)
actor_net.load_state_dict(torch.load(model_path))

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

curr_state = po_env.reset()
curr_state = tensor(curr_state).float().to(device)

render = True

obs = po_env.reset()
episode_rewards = []
episode_reward = 0
while len(episode_rewards) < num_eval_episodes:
    action, log_prob = actor_net(tensor(obs).view(1, -1, ).float().to(device))
    action = action.detach().to(cpu_device).numpy().squeeze()
    log_prob = log_prob.detach()
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
plt.plot()
