#!/usr/bin/env python3
import sys
import random
import os
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
import pomdp as pd
from math import exp
from statistics import mean
from scipy.stats import norm
from copy import deepcopy
from torchviz import make_dot

import tensorboard

def nn_equal(net1, net2):
    for p1, p2 in zip(net1.parameters(), net2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            # print(p1, p2)
            return False
    return True

def check_policies(policies):
    equal = True
    for i in range(len(policies) - 1):
        if not nn_equal(policies[i], policies[i + 1]):
            equal = False
    return equal 

class Actor(nn.Module):
    def __init__(self,  statespace_size, actionspace_size, hidden_size, learning_rate=3e-4, debug=False):
        super(Actor, self).__init__()
        self.actionspace_size = actionspace_size
        self.statespace_size = statespace_size
        self.debug = debug
        self.learning_rate = learning_rate

        self.actor_layer1 = nn.Linear(statespace_size, hidden_size * 2 )
        self.actor_layer2 = nn.Linear(hidden_size * 2, hidden_size) 
        self.actor_means = nn.Linear(hidden_size, actionspace_size)
        self.actor_devs = nn.Linear(hidden_size, actionspace_size)

    def forward(self,state):
        action_distribution = F.relu(self.actor_layer1(state))
        action_distribution = self.actor_layer2(action_distribution)

        means = self.actor_means(F.relu(action_distribution))
        log_deviations = self.actor_devs(F.relu(action_distribution))


        if self.debug:
            print("Means: {}".format(means))
            print("Log Deviations: {}".format(log_deviations))
        return means, log_deviations

 
class Critic(nn.Module):
    def __init__(self,  statespace_size, actionspace_size, hidden_size, learning_rate=3e-4, debug=False):
        super(Critic,self).__init__()
        self.actionspace_size = actionspace_size
        self.statespace_size = statespace_size
        self.debug = debug
        self.learning_rate = learning_rate

        self.critic_layer1 = nn.Linear(statespace_size, hidden_size * 2)
        self.critic_layer2 = nn.Linear(hidden_size * 2, hidden_size)
        self.critic_layer3 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        state_estimate = F.relu(self.critic_layer1(state))
        state_estimate = F.relu(self.critic_layer2(state_estimate))
        state_estimate = self.critic_layer3(state_estimate)
        if self.debug:
            print("State value estimate: {}".format(state_estimate))
        return state_estimate 


class PPO:
    def __init__(self,  statespace_size, actionspace_size, hidden_size, device, discount, clipping_epsilon, learning_rate=3e-4, debug=False, lambda_return = 0.95):

        self.device = device
        self.actor= Actor(statespace_size, actionspace_size, hidden_size, learning_rate=3e-4).to(device)
        self.critic = Critic(statespace_size, actionspace_size, hidden_size, learning_rate=3e-4).to(device)
        self.actionspace_size = actionspace_size
        self.statespace_size = statespace_size
        self.debug = debug
        self.discount_rate = discount

        self.lambda_return = lambda_return
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.previous_policy = None 
        self.previous_optimizer = None
        self.clipping_epsilon = clipping_epsilon
        self.average_reward = 0
        self.count = 0


    def actor_error(self, reward, state_value, next_state_value):
        error = (reward + (self.discount_rate*next_state_value) ) - state_value
        return error

    def critic_error(self, reward, state_value, timestep):
        self.average_reward = self.average_reward + (reward - self.average_reward)/timestep
        self.count += 1
        return state_value - self.average_reward

    def actor_loss(self, advantage, ratio):
        objective = torch.mul(advantage, ratio)
        clipped_objective = None
        if advantage >= advantage.new_zeros(advantage.size()):
            clipped_objective = torch.mul(advantage, (1.0 + self.clipping_epsilon))
        else:
            clipped_objective = torch.mul(advantage, (1.0 - self.clipping_epsilon))
        
        return torch.min(clipped_objective,torch.mean(objective))

def __main__():    

    environment_name = sys.argv[1]
    run_name_prefix = sys.argv[2]
    pomdp_type = sys.argv[3]
    seed = 42

    model_name = "PPO-{}-{}".format(environment_name, pomdp_type)
    run_name = "{}-{}".format(run_name_prefix, model_name)

    if not os.path.isdir('./models'):
        os.mkdir('./models')
    if not os.path.isdir('./models/{}'.format(run_name)):
        os.mkdir('./models/{}'.format(run_name))

    if not os.path.isdir('./runs'):
        os.mkdir('./runs')
    if not os.path.isdir('./runs/{}'.format(run_name)):
        os.mkdir('./runs/{}'.format(run_name))

    writer = SummaryWriter("./runs/{}/".format(run_name))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_size = 64 
    discount = 0.99
    clipping_epsilon = 0.2
    num_epochs = 10
    minibatch_size = 64
    env = gym.make(environment_name)
    pomdp = None

    if pomdp_type == 'faulty':
        if environment_name == 'Hopper-v2':
            pomdp = pd.PartiallyObservableEnv(env, seed, faulty=(20,env.observation_space.shape[0]))
        else:
            pomdp = pd.PartiallyObservableEnv(env, seed, faulty=(20,27))
            
    elif pomdp_type == 'noisy':
        pomdp = pd.PartiallyObservableEnv(env, seed, noisy=(0,0.1))
    else:
        pomdp = pd.PartiallyObservableEnv(env, seed)

    num_iterations = 10000
    if environment_name == 'Ant-v2':
        num_iterations = num_iterations * 3

    state = torch.tensor(pomdp.reset()).float().to(device)
    statespace_size = pomdp.env.observation_space.shape[0]
    actionspace_size = pomdp.env.action_space.shape[0]

    print("|S|: {}".format(statespace_size))
    print("|A|: {}".format(actionspace_size))

    horizon = 2048
    best_avg_reward = None
    agent = PPO(statespace_size, actionspace_size, hidden_size, device, discount, clipping_epsilon)
    
    agent.actor.train()
    agent.critic.train()

    for iteration in range(num_iterations):
        length = 0
        error_list = []
        ratio_list = []
        trajectory = []

        previous_reward = None
        previous_value = None
        error = None
        history = {}
               
        for step in range(horizon + 1):
            action_distribution = agent.actor(state)
            value = agent.critic(state)

            if previous_reward is not None:
                actor_error = agent.actor_error(previous_reward, previous_value, value)
                critic_error = agent.critic_error(previous_reward, previous_value, step)
                history['actor_error'] = actor_error
                history['critic_error'] = critic_error
                trajectory.append(history)

            means = action_distribution[0]
            log_deviations =  action_distribution[1]
            #log_deviations.register_hook(lambda grad: print(grad))
            deviations = torch.tensor([torch.exp(dev)**2 for dev in log_deviations]).to(device)
            action = Normal(means, deviations)
            history['policy_deviation'] = log_deviations

            next_state, reward, done, info = pomdp.step(action.sample().cpu())
            history['reward'] = reward

            if agent.previous_policy is not None:

                past_distribution = agent.previous_policy(state) 
                past_deviations = past_distribution[1].detach()
                history['previous_policy_deviation'] = past_deviations
            else:
                history['previous_policy_deviation'] = log_deviations.detach()

            previous_reward = reward
            previous_value = value
            state = torch.tensor(next_state).float().to(device)
            
        del trajectory[-1]
        old_actor_policy = deepcopy(agent.actor.state_dict())
        previous_policy = Actor(statespace_size, actionspace_size, hidden_size, learning_rate=3e-4).to(device)                
        previous_policy.load_state_dict(old_actor_policy)
        agent.previous_policy = previous_policy

        average_reward = agent.average_reward
        agent.average_reward = 0
        agent.count = 0
        if best_avg_reward == None or average_reward > best_average_reward:
            best_average_reward = average_reward
            writer.add_scalar('Reward/train', average_reward, iteration)
            torch.save(agent.actor.state_dict(), 'models/{}/best-{}-actor_net.pt'.format(run_name, model_name))
            torch.save(agent.critic.state_dict(), 'models/{}/best-{}-critic_net.pt'.format(run_name, model_name))

        critic_loss = torch.tensor([0],dtype=torch.float32,requires_grad=True).to(device)
        advantage = torch.tensor([0],dtype=torch.float32,requires_grad=True).to(device)
        actor_loss = torch.tensor([0],dtype=torch.float32,requires_grad=True).to(device)            
        timestep = 0

        losses = []

        for history in reversed(trajectory):
            timestep += 1
            loss_sample = {}
            advantage = advantage * agent.lambda_return * agent.discount_rate 
            ratio = history['policy_deviation'] - history['previous_policy_deviation']
            advantage += history['actor_error']
            actor_loss_t = agent.actor_loss(advantage, torch.exp(ratio))
            loss_sample['actor_loss'] = actor_loss_t
            actor_loss += actor_loss_t

            critic_loss += torch.mul(history['critic_error'], history['critic_error'])
            loss_sample['critic_loss'] = critic_loss
            losses.append(loss_sample)

        for epoch in range(num_epochs):

            minibatch = random.choices(losses, k=minibatch_size)
            batch_critic_loss = torch.tensor([0],dtype=torch.float32,requires_grad=True).to(device)
            batch_actor_loss = torch.tensor([0],dtype=torch.float32,requires_grad=True).to(device)            

            for sample in minibatch:
                batch_actor_loss += sample['actor_loss']
                batch_critic_loss += sample['critic_loss']

            batch_actor_loss = batch_actor_loss/ minibatch_size
            batch_critic_loss = batch_critic_loss / minibatch_size

            neg_actor_loss = -1 * batch_actor_loss

            agent.actor_optimizer.zero_grad()
            neg_actor_loss.backward(retain_graph=True)
            agent.actor_optimizer.step()
            writer.add_scalar('Loss/actor_net',neg_actor_loss.detach().cpu(), iteration)

            agent.critic_optimizer.zero_grad()
            batch_critic_loss.backward(retain_graph=True)
            agent.critic_optimizer.step()
            writer.add_scalar('Loss/critic_net', batch_critic_loss.detach().cpu(), iteration)

        
        if iteration % (1000) == 0 or iteration == num_iterations - 1:
            print("Reward for iteration {}: {}".format(iteration,average_reward, iteration))
            torch.save(agent.actor.state_dict(), 'models/{}/{}-actor_net-{}.pt'.format(run_name, model_name, iteration))
            torch.save(agent.critic.state_dict(), 'models/{}/{}-critic_net-{}.pt'.format(run_name, model_name, iteration))
            

if __name__ == '__main__':
    __main__()
