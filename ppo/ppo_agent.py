#!/usr/bin/env python3
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, debug=True statespace_size, actionspace_size, hidden_size, learning_rate=1e-5):
        super(ActorCritic,self).__init__()
        self.actionspace_size = actionspace_size
        self.statespace_size = statespace_size
        self.debug = debug

        self.actor_layer1 = nn.Linear(statespace_size, hidden_size )
        self.actor_layer2 = nn.Linear(hidden_size, num_actions)

        self.critic_layer1 = nn.Linear(statespace_size, hidden_size)
        self.critic_layer2 = nn.Linear(hidden_size, 1)


    def action_distribution(self, state):
        action_distribution = F.relu(self.actor_layer1(state))
        action_distribution = self.actor_layer2(state)
        if self.debug:
            print("Action distribution: {}".format(action))
        return action_distribution

    def state_estimate(self, state):
        state_estimate = F.relu(self.crtic_layer1(state))
        state_estimate = self.critic_layer2(value)
        if self.debug:
            print("State value estimate: {}".format(state_estimate))
        return state_estimate 

    def forward(self, state):
        if self.debug:
            print("Input state: {}".forrmat(state))

        state_estimate = self.state_estimate(state)
        action_distribution = self.action_distribution(state)

        action = F.softmax(F.relu(action_distribution))
        return action, value

class PPO:
    def __init__(self, statespace_size, actionspace_size, hidden_size, learning_rate=1e-5):

        self.actor_critic = ActorCritic(statespace_size, actionspace_size, hidden_size, learning_rate=1e-5)
        self.actionspace_size = actionspace_size
        self.statespace_size = statespace_size
        self.debug = debug

    def calculate_advantage(self):



def __main__():    
    env = gym.make('Ant-v2')

    state = env.reset()
    statespace_size = env.observation_space.shape[0]
    actionspace_size = env.action_space.shape[0]

    print("|S|: {}".format(statespace_size))
    print("|A|: {}".format(actionspace_size))

    episode_length = 500
    show_first = True

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

if __name__ == '__main__':
    __main__()
