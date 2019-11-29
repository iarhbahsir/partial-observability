import random
import sys
import os

import numpy as np
import matplotlib.pyplot as plt

from torch import tensor
from torch import cat
from torch import clamp
from torch.distributions import normal
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch

import gym

import pomdp

rand_seed = 42

environment_name = sys.argv[1]
model_name = "SAC-{}".format(environment_name)
run_name = "{}-{}".format(sys.argv[2], model_name)

if not os.path.isdir('./models'):
    os.mkdir('./models')
    os.mkdir('./models/{}'.format(run_name))
elif not os.path.isdir('./models/{}'.format(run_name)):
    os.mkdir('./models/{}'.format(run_name))

if not os.path.isdir('./runs'):
    os.mkdir('./runs')
    os.mkdir('./runs/{}'.format(run_name))
elif not os.path.isdir('./runs/{}'.format(run_name)):
    os.mkdir('./runs/{}'.format(run_name))


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

num_iterations = parameters['num_iterations'][environment_name]
learning_rate = parameters['learning_rate'][environment_name]
discount_rate = parameters['discount_rate'][environment_name]
replay_buffer_max_size = parameters['replay_buffer_max_size'][environment_name]
target_smoothing_coefficient = parameters['target_smoothing_coefficient'][environment_name]
target_update_interval = parameters['target_update_interval'][environment_name]
num_gradient_steps = parameters['num_gradient_steps'][environment_name]
num_env_steps = parameters['num_env_steps'][environment_name]
temperature = 1/parameters['reward_scale'][environment_name]
minibatch_size = parameters['minibatch_size'][environment_name]
STATE_DIM = parameters['state_dim'][environment_name]
ACTION_DIM = parameters['action_dim'][environment_name]
num_iter_before_train = parameters['num_iter_before_train'][environment_name]
hidden_layer_size = parameters['hidden_layer_size'][environment_name]

writer = SummaryWriter(log_dir="./runs/{}/".format(run_name))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu_device = torch.device("cpu")


# define actor network
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


# define critic network
class SACCriticNN(nn.Module):
    def __init__(self):
        super(SACCriticNN, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM + ACTION_DIM, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, ACTION_DIM)

    def forward(self, x_state, x_action):
        x = cat((x_state, x_action), dim=1)  # concatenate inputs along 0th dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# define soft state value network
class SACStateValueNN(nn.Module):
    def __init__(self):
        super(SACStateValueNN, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, 1)

    def forward(self, x_state):
        x = F.relu(self.fc1(x_state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Initialize parameter vectors ψ, ψ¯, θ, φ.
state_value_net = SACStateValueNN().to(device)
state_value_target_net = SACStateValueNN().to(device)
critic_net_1 = SACCriticNN().to(device)
critic_net_2 = SACCriticNN().to(device)
actor_net = SACActorNN().to(device)

# make the state value target net parameters the same
state_value_target_net.load_state_dict(state_value_net.state_dict())

# initialize replay buffer D
replay_buffer = []

# initialize train and test environments
env = gym.make(environment_name)

if sys.argv[3] == 'faulty':
    if environment_name == 'Hopper-v2':
        po_env = pomdp.PartiallyObservableEnv(env, rand_seed, faulty=(20, env.observation_space.shape[0]))
    elif environment_name == 'Ant-v2':
        po_env = pomdp.PartiallyObservableEnv(env, rand_seed, faulty=(20, 27))
elif sys.argv[3] == 'noisy':
    po_env = pomdp.PartiallyObservableEnv(env, rand_seed, noisy=(0, 0.1))
else:
    po_env = pomdp.PartiallyObservableEnv(env, rand_seed)

curr_state = po_env.reset()
curr_state = tensor(curr_state).float().to(device)

test_env = gym.make(environment_name)

if sys.argv[3] == 'faulty':
    if environment_name == 'Hopper-v2':
        test_po_env = pomdp.PartiallyObservableEnv(test_env, rand_seed, faulty=(20, env.observation_space.shape[0]))
    elif environment_name == 'Ant-v2':
        test_po_env = pomdp.PartiallyObservableEnv(test_env, rand_seed, faulty=(20, 27))
elif sys.argv[3] == 'noisy':
    test_po_env = pomdp.PartiallyObservableEnv(test_env, rand_seed, noisy=(0, 0.1))
else:
    test_po_env = pomdp.PartiallyObservableEnv(test_env, rand_seed)

curr_test_state = test_po_env.reset()
greatest_avg_episode_rewards = -np.inf

# initialize optimizers for each network except target (parameters updated manually)
state_value_net_optimizer = optim.Adam(state_value_net.parameters(), lr=learning_rate)
critic_net_1_optimizer = optim.Adam(critic_net_1.parameters(), lr=learning_rate)
critic_net_2_optimizer = optim.Adam(critic_net_2.parameters(), lr=learning_rate)
actor_net_optimizer = optim.Adam(actor_net.parameters(), lr=learning_rate)

# for each iteration do
for t in range(num_iterations):
    # for each environment step do
    # (in practice, at most one env step per gradient step)
    # at ∼ πφ(at|st)
    if t > num_iter_before_train:
        action, log_prob = actor_net(curr_state.view(1, -1,).float().to(device))
    else:
        action = tensor(env.action_space.sample()).float().to(device)
        log_prob = torch.ones(action.shape)
    action_np = action.detach().to(cpu_device).numpy().squeeze()
    log_prob = log_prob.detach()

    # st+1 ∼ p(st+1|st, at)
    next_state, reward, done, _ = po_env.step(action_np)

    # D ← D ∪ {(st, at, r(st, at), st+1)}
    replay_buffer.append((curr_state.to(cpu_device).view(1, -1, ).float(), tensor(action_np).to(cpu_device).float().view(1, -1, ), log_prob.float().to(cpu_device).view(1, -1, ),
                          tensor(reward).float().to(cpu_device).view(1, 1, ), tensor(next_state).float().to(cpu_device).view(1, -1, ),
                          tensor(done).to(cpu_device).view(1, 1, ).float()))
    if len(replay_buffer) > replay_buffer_max_size + 10:
        replay_buffer = replay_buffer[10:]

    del action, log_prob

    if t > num_iter_before_train:
    # for each gradient step do
        for gradient_step in range(num_gradient_steps):
            # Sample mini-batch of N transitions (s, a, r, s') from D
            transitions_minibatch = random.choices(replay_buffer, k=minibatch_size)
            minibatch_states, minibatch_actions, minibatch_action_log_probs, minibatch_rewards, minibatch_next_states, minibatch_dones = [cat(mb, dim=0).to(device) for mb in zip(*transitions_minibatch)]
            minibatch_states = minibatch_states.float()

            # ψ ← ψ − λV ∇ˆψJV (ψ)
            state_value_net.zero_grad()
            minibatch_actions_new, minibatch_action_log_probs_new = actor_net(minibatch_states)
            state_value_net_loss = torch.mean(0.5 * (state_value_net(minibatch_states) - (torch.min(critic_net_1(minibatch_states, minibatch_actions_new), critic_net_2(minibatch_states, minibatch_actions_new)) - torch.mul(temperature, minibatch_action_log_probs_new))) ** 2)
            state_value_net_optimizer.zero_grad()
            state_value_net_loss.backward()
            state_value_net_optimizer.step()
            writer.add_scalar('Loss/state_value_net', state_value_net_loss.detach().to(cpu_device).numpy().squeeze(), t)

            del state_value_net_loss, minibatch_actions_new, minibatch_action_log_probs_new

            # θi ← θi − λQ∇ˆθiJQ(θi) for i ∈ {1, 2}
            critic_net_1.zero_grad()
            critic_net_1_loss = torch.mean(0.5 * (critic_net_1(minibatch_states, minibatch_actions) - (minibatch_rewards + discount_rate*state_value_target_net(minibatch_next_states)*(-minibatch_dones + 1))) ** 2)
            critic_net_1_optimizer.zero_grad()
            critic_net_1_loss.backward()
            critic_net_1_optimizer.step()
            writer.add_scalar('Loss/critic_net_1', critic_net_1_loss.detach().to(cpu_device).numpy().squeeze(), t)

            del critic_net_1_loss

            critic_net_2.zero_grad()
            critic_net_2_loss = torch.mean(0.5 * (critic_net_2(minibatch_states, minibatch_actions) - (minibatch_rewards + discount_rate*state_value_target_net(minibatch_next_states)*(-minibatch_dones + 1))) ** 2)
            critic_net_2_optimizer.zero_grad()
            critic_net_2_loss.backward()
            critic_net_2_optimizer.step()
            writer.add_scalar('Loss/critic_net_2', critic_net_2_loss.detach().to(cpu_device).numpy().squeeze(), t)

            del critic_net_2_loss

            # φ ← φ − λπ∇ˆφJπ(φ)
            actor_net.zero_grad()
            minibatch_actions_new, minibatch_action_log_probs_new = actor_net(minibatch_states)
            actor_net_loss = torch.mean(torch.mul(minibatch_action_log_probs_new, temperature) - torch.min(critic_net_1(minibatch_states, minibatch_actions_new), critic_net_2(minibatch_states, minibatch_actions_new)))
            actor_net_optimizer.zero_grad()
            actor_net_loss.backward()
            actor_net_optimizer.step()
            writer.add_scalar('Loss/actor_net', actor_net_loss.detach().to(cpu_device).numpy().squeeze(), t)

            del actor_net_loss, minibatch_actions_new, minibatch_action_log_probs_new

            # ψ¯ ← τψ + (1 − τ )ψ¯
            for state_value_target_net_parameter, state_value_net_parameter in zip(state_value_target_net.parameters(), state_value_net.parameters()):
                state_value_target_net_parameter.data = target_smoothing_coefficient*state_value_net_parameter + (1 - target_smoothing_coefficient)*state_value_target_net_parameter
            # end for

            del minibatch_states, minibatch_actions, minibatch_action_log_probs, minibatch_rewards, minibatch_next_states, minibatch_dones

    if t % 1000 == 0 or t == num_iterations - 1:
        print("iter", t)
        torch.save(state_value_net.state_dict(), 'models/{}/{}-state_value_net.pt'.format(run_name, model_name))
        torch.save(state_value_target_net.state_dict(), 'models/{}/{}-state_value_target_net.pt'.format(run_name, model_name))
        torch.save(critic_net_1.state_dict(), 'models/{}/{}-critic_net_1.pt'.format(run_name, model_name))
        torch.save(critic_net_2.state_dict(), 'models/{}/{}-critic_net_2.pt'.format(run_name, model_name))
        torch.save(actor_net.state_dict(), 'models/{}/{}-actor_net.pt'.format(run_name, model_name))

    if not done:
        curr_state = tensor(next_state).float().to(device)
    else:
        curr_state = po_env.reset()
        curr_state = tensor(curr_state).float().to(device)

    if t % (num_iterations // 1000) == 0 or t == num_iterations - 1:
        render = False
        num_eval_episodes = 10

        test_obs = test_po_env.reset()
        episode_rewards = []
        episode_reward = 0
        while len(episode_rewards) < num_eval_episodes:
            test_action, test_action_log_prob = actor_net(tensor(test_obs).view(1, -1, ).float().to(device))
            test_action = test_action.detach().to(cpu_device).numpy().squeeze()
            test_obs, test_reward, test_done, _ = test_po_env.step(test_action)
            episode_reward += test_reward
            if test_done:
                episode_rewards.append(episode_reward)
                episode_reward = 0
                test_obs = test_po_env.reset()
            if render:
                test_po_env.render()

        avg_episode_rewards = np.mean(np.asarray(episode_rewards))
        writer.add_scalar('Reward/test', avg_episode_rewards, t)
        if avg_episode_rewards > greatest_avg_episode_rewards:
            torch.save(actor_net.state_dict(), 'models/{}/best-{}-actor_net.pt'.format(run_name, model_name))
            greatest_avg_episode_rewards = avg_episode_rewards

    if t % (num_iterations // 4) == 0 or t == num_iterations - 1:
        torch.save(actor_net.state_dict(), 'models/{}/{}-actor_net-{}.pt'.format(run_name, model_name, t))


# end for

render = False
num_eval_episodes = 40

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
plt.savefig("models/{}/episode_rewards_hist.png".format(run_name))
print("Mean total episode reward:", np.mean(episode_rewards))