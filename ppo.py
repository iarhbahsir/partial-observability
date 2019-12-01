#!/usr/bin/env python3
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pomdp as pd
from math import exp
from statistics import mean
from scipy.stats import norm

class ActorCritic(nn.Module):
    def __init__(self,  statespace_size, actionspace_size, hidden_size, learning_rate=1e-5, debug=False):
        super(ActorCritic,self).__init__()
        self.actionspace_size = actionspace_size
        self.statespace_size = statespace_size
        self.debug = debug
        self.learning_rate = learning_rate

        self.actor_layer1 = nn.Linear(statespace_size, hidden_size * 2 )
        self.actor_layer2 = nn.Linear(hidden_size * 2, hidden_size)

        self.actor_means = nn.Linear(hidden_size, actionspace_size)
        self.actor_devs = nn.Linear(hidden_size, actionspace_size)
        self.critic_layer1 = nn.Linear(statespace_size, hidden_size * 2)
        self.critic_layer2 = nn.Linear(hidden_size * 2, hidden_size)
        self.critic_layer3 = nn.Linear(hidden_size, 1)


    def action_distribution(self, state):
        action_distribution = F.relu(self.actor_layer1(state))
        action_distribution = self.actor_layer2(action_distribution)

        means = self.actor_means(F.relu(action_distribution))
        log_deviations = self.actor_devs(F.relu(action_distribution))


        if self.debug:
            print("Means: {}".format(means))
            print("Log Deviations: {}".format(log_deviations))
        return means, log_deviations

    def state_estimate(self, state):
        state_estimate = F.relu(self.critic_layer1(state))
        state_estimate = F.relu(self.critic_layer2(state_estimate))
        state_estiamte = self.critic_layer3(state_estimate)
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
    def __init__(self,  statespace_size, actionspace_size, hidden_size, device, discount, learning_rate=1e-5, debug=False, lambda_return = 0.95):

        self.device = device
        self.actor_critic = ActorCritic(statespace_size, actionspace_size, hidden_size, learning_rate=1e-5).to(device)
        self.actionspace_size = actionspace_size
        self.statespace_size = statespace_size
        self.debug = debug
        self.discount_rate = discount

        self.previous_policy = None 
        self.lambda_return = lambda_return


    def error(self, reward, state_value, next_state_value):
        err = (reward + (self.discount_rate*next_state_value) ) - state_value
        return err




def __main__():    
    cpu_device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_size = 256
    discount = 0.99
    clipping_epsilon = 0.2
    env = gym.make('Ant-v2')
    # Let's start simple and do the MDP before we get the rest of this 
    pomdp = pd.PartiallyObservableEnv(env, seed=42)
    num_iterations = 2

    state = torch.tensor(pomdp.reset()).float().to(device)
    statespace_size = pomdp.env.observation_space.shape[0]
    actionspace_size = pomdp.env.action_space.shape[0]

    print("|S|: {}".format(statespace_size))
    print("|A|: {}".format(actionspace_size))

    horizon = 500
    show_first = True

    agent = PPO(statespace_size, actionspace_size, hidden_size, device, discount)

    
    for i in range(num_iterations):
        distributions = []
        length = 0
        timestep = {}
        error_list = []
        ratio_list = []

        previous_reward = None
        previous_value = None
        error = None

        for step in range(horizon):
            action_distribution, value = agent.actor_critic.forward(state)

            if previous_reward is not None:
                error = agent.error(previous_reward, previous_value, value)
                error_list.append(error.detach()[0])


            means = action_distribution[0].detach().cpu().numpy()
            log_deviations =  action_distribution[1].detach().cpu().numpy()
            deviations = [exp(dev)**2 for dev in log_deviations]
            distributions.append([means, log_deviations])
            action = np.random.normal(means, deviations , actionspace_size)

            next_state, reward, done, info = pomdp.step(action)

            ratio = None
            if agent.previous_policy is not None:
                
                past_distribution, _ = agent.previous_policy.forward(state) 
                past_deviations = past_distribution[1].detach().cpu().numpy()

                ratio = log_deviations - past_deviations
                ratio = [exp(item) for item in ratio]
            else:
                ratio = [1 for i in range(len(log_deviations))]
                
            
            clipped_ratio = np.clip(ratio,  a_min=(1 - clipping_epsilon), a_max=(1+clipping_epsilon))
            ratio = [min(ratio[i],clipped_ratio[i]) for i in range(len(ratio))]
                
            previous_reward = reward
            previous_value = value

            timestep['state'] = state
            timestep['next_state'] = next_state
            timestep['action_distro'] = action_distribution
            timestep['reward'] = reward
            timestep['error'] = error
            state = torch.tensor(next_state).float().to(device)
            ratio_list.append(ratio)


        agent.previous_policy = agent.actor_critic
        reversed_advantages = []
        advantage = 0
        for backwards_step in reversed(error_list):
            advantage = advantage * agent.lambda_return * agent.discount_rate 
            advantage += backwards_step
            reversed_advantages.append(advantage)

        reversed_advantages = [item.cpu().numpy() for item in reversed_advantages]
        #for i in range(len(reversed_advantages)):
        #    print(reversed_advantages[i])

        loss_list = []
        actor_loss = None

        for i in range(len(reversed_advantages)):
           advantage = reversed_advantages[len(reversed_advantages) - i - 1]
           print(advantage)
           ratio = ratio_list[i]
           loss_list.append(advantage * ratio)

        actor_loss = sum(loss_list)
        print(actor_loss)
        actor_loss = mean(actor_loss)
        print(actor_loss)
        neg_actor_loss = -1 * actor_loss

        #print([item for item in advantages])
            
         
        

    '''
    or i in range(episode_length):
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
