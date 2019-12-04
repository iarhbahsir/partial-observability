import random
import sys
import os

import numpy as np
import matplotlib.pyplot as plt

from torch import tensor
from torch import cat
from torch import clamp
from torch.distributions import normal, multivariate_normal
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch

import gym

import pomdp

rand_seed = 42

environment_name = sys.argv[1]
run_name_prefix = sys.argv[2]
pomdp_type = sys.argv[3]

model_name = "TD3-{}-{}".format(environment_name, pomdp_type)
run_name = "{}-{}".format(run_name_prefix, model_name)

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


num_iterations = parameters['num_iterations'][environment_name]
replay_memory_max_size = parameters['replay_memory_max_size'][environment_name]
sigma = parameters['sigma'][environment_name]
smoothing_sigma = parameters['smoothing_sigma'][environment_name]
minibatch_size = parameters['minibatch_size'][environment_name]
discount_rate = parameters['discount_rate'][environment_name]
steps_until_policy_update = parameters['steps_until_policy_update'][environment_name]
target_update_ratio = parameters['target_update_ratio'][environment_name]
epsilon_limit = parameters['epsilon_limit'][environment_name]
state_dim = parameters['state_dim'][environment_name]
action_dim = parameters['action_dim'][environment_name]
num_iter_before_train = parameters['num_iter_before_train'][environment_name]

writer = SummaryWriter(log_dir="./runs/{}/".format(run_name))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu_device = torch.device("cpu")


# define actor network
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


# define critic network
class TD3LunarLanderContinuousCriticNN(nn.Module):
    def __init__(self):
        super(TD3LunarLanderContinuousCriticNN, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = cat((state, action), dim=1)  # concatenate inputs along 0th dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Initialize critic networks Qθ1, Qθ2, and actor network πφ with random parameters θ1, θ2, φ
critic_net_1 = TD3LunarLanderContinuousCriticNN()
critic_net_2 = TD3LunarLanderContinuousCriticNN()
actor_net = TD3LunarLanderContinuousActorNN()

# Initialize target networks θ'1 ← θ1, 0'2 ← θ2, φ' ← φ
critic_target_net_1 = TD3LunarLanderContinuousCriticNN()
critic_target_net_1.load_state_dict(critic_net_1.state_dict())
critic_target_net_2 = TD3LunarLanderContinuousCriticNN()
critic_target_net_2.load_state_dict(critic_net_2.state_dict())
actor_target_net = TD3LunarLanderContinuousActorNN()
actor_target_net.load_state_dict(actor_net.state_dict())

# make GPU compatible
critic_net_1 = critic_net_1.to(device)
critic_net_2 = critic_net_2.to(device)
critic_target_net_1 = critic_target_net_1.to(device)
critic_target_net_2 = critic_target_net_2.to(device)
actor_net = actor_net.to(device)
actor_target_net = actor_target_net.to(device)

# Initialize replay buffer B
replay_buffer = []

# initialize the environments
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
min_action = torch.from_numpy(env.action_space.low).to(device)
max_action = torch.from_numpy(env.action_space.high).to(device)

test_env = gym.make(environment_name)

if pomdp_type == 'faulty':
    if environment_name == 'Hopper-v2':
        test_po_env = pomdp.PartiallyObservableEnv(test_env, rand_seed, faulty=(20, env.observation_space.shape[0]))
    elif environment_name == 'Ant-v2':
        test_po_env = pomdp.PartiallyObservableEnv(test_env, rand_seed, faulty=(20, 27))
elif pomdp_type == 'noisy':
    test_po_env = pomdp.PartiallyObservableEnv(test_env, rand_seed, noisy=(0, 0.1))
else:
    test_po_env = pomdp.PartiallyObservableEnv(test_env, rand_seed)

curr_test_state = test_po_env.reset()
greatest_avg_episode_rewards = -np.inf

# initialize critic losses
critic_net_1_loss_fn = nn.MSELoss(reduction='mean')
critic_net_2_loss_fn = nn.MSELoss(reduction='mean')

# initialize optimizers
critic_net_1_optimizer = optim.Adam(critic_net_1.parameters(), lr=0.001)
critic_net_2_optimizer = optim.Adam(critic_net_2.parameters(), lr=0.001)
actor_net_optimizer = optim.Adam(actor_net.parameters(), lr=0.001)

# initialize normal distribution N
normal_dist = normal.Normal(0, sigma)
smoothing_normal_dist = normal.Normal(0, smoothing_sigma)

# for t = 1 to T do
for t in range(num_iterations):
    # Select action with exploration noise a ∼ πφ(s) + ϵ ,ϵ ∼ N (0, σ), and observe reward r and new state s'
    if t > num_iter_before_train:
        action = torch.max(torch.min(actor_net(curr_state.view(1, -1, ).float()) + clamp(normal_dist.sample(sample_shape=(action_dim,)).to(device), -epsilon_limit, epsilon_limit), max_action), min_action)
        action = action.detach().to(cpu_device).numpy().squeeze()
    else:
        action = env.action_space.sample()
        action = action.squeeze()
    next_state, reward, done, _ = po_env.step(action)

    # Store transition tuple (s, a, r, s') in B
    replay_buffer.append((curr_state.view(1, -1,), tensor(action).to(device).view(1, -1,), tensor(reward).float().to(device).view(1, 1,), tensor(next_state).to(device).view(1, -1,), tensor(done).to(device).view(1, 1,)))
    if len(replay_buffer) > replay_memory_max_size + 10:
        replay_buffer = replay_buffer[10:]

    if t > num_iter_before_train:
        # Sample mini-batch of N transitions (s, a, r, s') from B
        transitions_minibatch = random.choices(replay_buffer, k=minibatch_size)
        minibatch_states, minibatch_actions, minibatch_rewards, minibatch_next_states, minibatch_dones = [cat(mb, dim=0) for mb in zip(*transitions_minibatch)]
        minibatch_states = minibatch_states.float()
        minibatch_next_states = minibatch_next_states.float()

        # a˜ ← πφ0 (s') + ϵ, ϵ ∼ clip(N (0, σ˜), −c, c)
        sampled = smoothing_normal_dist.sample(sample_shape=(minibatch_size, action_dim,)).to(device)
        minibatch_next_actions = torch.max(torch.min(actor_target_net(minibatch_next_states) + clamp(sampled, -epsilon_limit, epsilon_limit), max_action), min_action)

        # y ← r + γ mini=1,2 Qθ'i(s', a˜)
        minibatch_y = minibatch_rewards + discount_rate * torch.min(critic_target_net_1(minibatch_next_states, minibatch_next_actions), critic_target_net_2(minibatch_next_states, minibatch_next_actions)) * (-minibatch_dones.float() + 1)

        # Update critics θi ← argminθi (N^−1)*Σ(y−Qθi(s, a))^2
        critic_net_1.zero_grad()
        critic_net_1_loss = critic_net_1_loss_fn(critic_net_1(minibatch_states, minibatch_actions), minibatch_y)
        critic_net_1_loss.backward(retain_graph=True)
        critic_net_1_optimizer.step()
        writer.add_scalar('Loss/critic_net_1', critic_net_1_loss.detach().to(cpu_device).numpy().squeeze(), t)

        critic_net_2.zero_grad()
        critic_net_2_loss = critic_net_2_loss_fn(critic_net_2(minibatch_states, minibatch_actions), minibatch_y)
        critic_net_2_loss.backward(retain_graph=True)
        critic_net_2_optimizer.step()
        writer.add_scalar('Loss/critic_net_2', critic_net_2_loss.detach().to(cpu_device).numpy().squeeze(), t)

        # if t mod d then
        if t % steps_until_policy_update == 0:
            # Update φ by the deterministic policy gradient: ∇φJ(φ) = N −1 P∇aQθ1(s, a)|a=πφ(s)∇φπφ(s)
            actor_net.zero_grad()
            actor_net_loss = -1 * critic_net_1(minibatch_states, actor_net(minibatch_states)).mean()  # gradient ascent
            actor_net_loss.backward()
            actor_net_optimizer.step()
            writer.add_scalar('Loss/actor_net', actor_net_loss.detach().to(cpu_device).numpy().squeeze(), t)

            # Update target networks:
            # θ'i ← τθi + (1 − τ )θ'i
            for critic_target_net_1_parameter, critic_net_1_parameter in zip(critic_target_net_1.parameters(), critic_net_1.parameters()):
                critic_target_net_1_parameter.data = target_update_ratio*critic_net_1_parameter + (1-target_update_ratio)*critic_target_net_1_parameter

            for critic_target_net_2_parameter, critic_net_2_parameter in zip(critic_target_net_2.parameters(), critic_net_2.parameters()):
                critic_target_net_2_parameter.data = target_update_ratio*critic_net_2_parameter + (1-target_update_ratio)*critic_target_net_2_parameter

            # φ' ← τφ + (1 − τ )φ'
            for actor_target_net_parameter, actor_net_parameter in zip(actor_target_net.parameters(), actor_net.parameters()):
                actor_target_net_parameter.data = target_update_ratio*actor_net_parameter + (1-target_update_ratio)*actor_target_net_parameter

    # end if

    if t % (num_iterations // 1000) == 0 or t == num_iterations - 1:
        print("iter", t)
        torch.save(critic_net_1.state_dict(), 'models/{}/{}-critic_net_1.pt'.format(run_name, model_name))
        torch.save(critic_target_net_1.state_dict(), 'models/{}/{}-critic_target_net_1.pt'.format(run_name, model_name))
        torch.save(critic_net_2.state_dict(), 'models/{}/{}-critic_net_2.pt'.format(run_name, model_name))
        torch.save(critic_target_net_2.state_dict(), 'models/{}/{}-critic_target_net_2.pt'.format(run_name, model_name))
        torch.save(actor_net.state_dict(), 'models/{}/{}-actor_net.pt'.format(run_name, model_name))
        torch.save(actor_target_net.state_dict(), 'models/{}/{}-actor_target_net.pt'.format(run_name, model_name))

    if not done:
        curr_state = tensor(next_state).float().to(device)
    else:
        curr_state = po_env.reset()
        curr_state = tensor(curr_state).float().to(device)

    if t % (num_iterations // 25) == 0 or t == num_iterations - 1:
        render = False
        num_eval_episodes = 10

        test_obs = test_po_env.reset()
        episode_rewards = []
        episode_reward = 0
        while len(episode_rewards) < num_eval_episodes:
            test_action = actor_target_net(tensor(test_obs).float().to(device)).detach().to(cpu_device).numpy().squeeze()
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
            torch.save(actor_target_net.state_dict(), 'models/{}/best-{}-actor_target_net.pt'.format(run_name, model_name))
            greatest_avg_episode_rewards = avg_episode_rewards

    if t % (num_iterations // 4) == 0 or t == num_iterations - 1:
        torch.save(actor_target_net.state_dict(), 'models/{}/{}-actor_target_net-{}.pt'.format(run_name, model_name, t))

# end for

render = False
num_eval_episodes = 40

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
plt.savefig("models/{}/episode_rewards_hist.png".format(run_name))
print("Mean total episode reward:", np.mean(episode_rewards))