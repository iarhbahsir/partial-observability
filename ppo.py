#!/usr/bin/env python3
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pomdp as pd

class ActorCritic(nn.Module):
    def __init__(self,  statespace_size, actionspace_size, hidden_size, learning_rate=1e-5, debug=True):
        super(ActorCritic,self).__init__()
        self.actionspace_size = actionspace_size
        self.statespace_size = statespace_size
        self.debug = debug
        self.learning_rate = learning_rate

        self.actor_layer1 = nn.Linear(statespace_size, hidden_size )
        self.actor_layer2 = nn.Linear(hidden_size, actionspace_size)

        self.critic_layer1 = nn.Linear(statespace_size, hidden_size)
        self.critic_layer2 = nn.Linear(hidden_size, 1)


    def action_distribution(self, state):
        action_distribution = F.relu(self.actor_layer1(state))
        action_distribution = self.actor_layer2(action_distribution)
        if self.debug:
            print("Action distribution: {}".format(action_distribution))
        return action_distribution

    def state_estimate(self, state):
        state_estimate = F.relu(self.critic_layer1(state))
        state_estimate = self.critic_layer2(state_estimate)
        if self.debug:
            print("State value estimate: {}".format(state_estimate))
        return state_estimate 

    def forward(self, state):
        if self.debug:
            print("Input state: {}".format(state))

        state_estimate = self.state_estimate(state)
        action_distribution = self.action_distribution(state)

        return action_distribution, state_estimate

class PPO:
    def __init__(self,  statespace_size, actionspace_size, hidden_size, device, learning_rate=1e-5, debug=True, gamma = 0.95):

        self.device = device
        self.actor_critic = ActorCritic(statespace_size, actionspace_size, hidden_size, learning_rate=1e-5).to(device)
        self.actionspace_size = actionspace_size
        self.statespace_size = statespace_size
        self.debug = debug

        self.current_policy = None
        self.previous_policy = None
        self.gamma = gamma






def __main__():    
    cpu_device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_size = 256
    env = gym.make('Ant-v2')
    # Let's start simple and do the MDP before we get the rest of this 
    pomdp = pd.PartiallyObservableEnv(env, seed=42)
    num_iterations = 1


    #TODO Do we need this?
    state = torch.tensor(pomdp.reset()).float().to(device)
    statespace_size = pomdp.env.observation_space.shape[0]
    actionspace_size = pomdp.env.action_space.shape[0]

    print("|S|: {}".format(statespace_size))
    print("|A|: {}".format(actionspace_size))

    episode_length = 500
    show_first = True

    agent = PPO(statespace_size, actionspace_size, hidden_size, device)

    
    for i in range(num_iterations):
        memory_dict = {}

        while True:
            action_distribution, value = agent.actor_critic.forward(state)

            action = F.softmax(action_distribution)
            print(action)
            action = action.detach().cpu()

            next_state, reward, done, info = pomdp.step(action)
            memory_dict['state'] = state
            memory_dict['next_state'] = next_state
            memory_dict['action_distro'] = action_distribution
            memory_dict['reward'] = reward


            if done:
                break
            print(action_distribution, value)

    '''

    for i in range(episode_length):
        env.render()
        action = np.random.randn(actionspace_size,1)
        if show_first:
            print("Action unedited: {}".format(action))

        action = action.reshape((1,-1)).astype(np.float32)
        if show_first:
            print("Action reshaped: {}".format(action))
        action_input = np.squeeze(action,axis=0)
        if show_first:
            print("Action moved again???? {}".format(action_input))
            show_first = False
        state, reward, done, info = env.step(action_input)
    '''

if __name__ == '__main__':
    __main__()
